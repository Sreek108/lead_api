"""
Geographical Analysis Engine
Analyzes lead distribution and performance by geography
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any, List
import logging
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeographicalAnalysisEngine:
    """
    Geographical Analysis Engine for Lead Management
    Provides market insights, country performance, and regional analysis
    """
    
    def __init__(self, server: str, database: str, username: str, password: str):
        """Initialize with database connection"""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        
    def connect(self) -> bool:
        """Establish database connection with retry logic and connection pooling"""
        try:
            conn_string = f"mssql+pymssql://{self.username}:{quote_plus(self.password)}@{self.server}/{self.database}?charset=utf8"
            
            # Create engine with connection pooling
            self.engine = create_engine(
                conn_string,
                pool_pre_ping=True,      # Test connections before using
                pool_recycle=3600,       # Recycle connections after 1 hour
                pool_size=5,             # Keep 5 connections in pool
                max_overflow=10,         # Allow 10 additional connections
                connect_args={
                    'timeout': 30,        # 30 second connection timeout
                    'login_timeout': 30   # 30 second login timeout
                },
                echo=False               # Don't log SQL queries
            )
            
            # Test connection immediately
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established successfully with connection pooling")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def discover_schema(self, table_name: str) -> List[str]:
        """Auto-discover table columns"""
        try:
            query = f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
            ORDER BY ORDINAL_POSITION
            """
            cols = pd.read_sql(query, self.engine)
            return cols['COLUMN_NAME'].tolist() if not cols.empty else []
        except Exception as e:
            logger.error(f"Schema discovery failed for {table_name}: {e}")
            return []
    
    def load_geographical_data(self) -> pd.DataFrame:
        """Load all geographical data with auto-discovery"""
        try:
            logger.info("Loading geographical data...")
            
            # Discover schemas
            lead_columns = self.discover_schema('Lead')
            country_columns = self.discover_schema('Country')
            cityregion_columns = self.discover_schema('CityRegion')
            
            # Check what's available
            has_country = 'CountryId' in lead_columns
            has_city_region = 'CityRegionId' in lead_columns
            
            if not has_country:
                raise ValueError("CountryId not found in Lead table")
            
            # Load leads with geography
            select_cols = ['LeadId', 'LeadCode', 'LeadStatusId', 'CreatedOn', 'CountryId']
            if has_city_region:
                select_cols.append('CityRegionId')
            
            leads_query = f"""
            SELECT {', '.join(select_cols)}
            FROM dbo.Lead
            WHERE IsActive = 1
            """
            
            leads = pd.read_sql(leads_query, self.engine)
            
            # Load countries
            countries = pd.read_sql("SELECT * FROM dbo.Country", self.engine)
            
            # Find country name column
            country_name_col = None
            for col in country_columns:
                if 'name' in col.lower() and ('_e' in col.lower() or 'english' in col.lower()):
                    country_name_col = col
                    break
            if not country_name_col:
                country_name_col = [c for c in country_columns if 'name' in c.lower()][0]
            
            countries = countries.rename(columns={country_name_col: 'CountryName'})
            
            # Find country code if exists
            if 'CountryCode' in country_columns:
                pass  # Already there
            elif any('ISO' in str(col).upper() for col in country_columns):
                code_col = [c for c in country_columns if 'ISO' in c.upper()][0]
                countries = countries.rename(columns={code_col: 'CountryCode'})
            else:
                countries['CountryCode'] = ''
            
            # Merge
            geo_data = leads.merge(
                countries[['CountryId', 'CountryName', 'CountryCode']], 
                on='CountryId', 
                how='left'
            )
            
            # Load city regions if available
            if has_city_region and cityregion_columns:
                city_regions = pd.read_sql("SELECT * FROM dbo.CityRegion", self.engine)
                
                region_name_col = None
                for col in cityregion_columns:
                    if 'name' in col.lower() and ('_e' in col.lower() or 'english' in col.lower()):
                        region_name_col = col
                        break
                if not region_name_col:
                    region_name_col = [c for c in cityregion_columns if 'name' in c.lower()][0]
                
                city_regions = city_regions.rename(columns={region_name_col: 'CityRegionName'})
                
                geo_data = geo_data.merge(
                    city_regions[['CityRegionId', 'CityRegionName']], 
                    on='CityRegionId', 
                    how='left'
                )
            
            # Load engagement data
            meetings = pd.read_sql("SELECT LeadId, StartDateTime FROM dbo.AgentMeetingAssignment", self.engine)
            calls = pd.read_sql("SELECT LeadId, CallDateTime FROM dbo.LeadCallRecord", self.engine)
            
            # Add meeting counts
            meeting_counts = meetings.groupby('LeadId').size().reset_index(name='MeetingCount')
            geo_data = geo_data.merge(meeting_counts, on='LeadId', how='left')
            geo_data['MeetingCount'] = geo_data['MeetingCount'].fillna(0)
            
            # Add call counts
            call_counts = calls.groupby('LeadId').size().reset_index(name='CallCount')
            geo_data = geo_data.merge(call_counts, on='LeadId', how='left')
            geo_data['CallCount'] = geo_data['CallCount'].fillna(0)
            
            # Calculate lead age
            geo_data['CreatedOn'] = pd.to_datetime(geo_data['CreatedOn'])
            geo_data['LeadAge_Days'] = (datetime.now() - geo_data['CreatedOn']).dt.days
            
            # Calculate basic scores
            geo_data['EngagementScore'] = (
                (geo_data['MeetingCount'] * 30) +
                (geo_data['CallCount'] * 10)
            )
            
            geo_data['FreshnessScore'] = geo_data['LeadAge_Days'].apply(lambda x: max(0, 100 - x))
            
            geo_data['LeadScore'] = (
                (geo_data['EngagementScore'] * 0.6) +
                (geo_data['FreshnessScore'] * 0.4)
            )
            
            if geo_data['LeadScore'].max() > 0:
                geo_data['LeadScore'] = (geo_data['LeadScore'] / geo_data['LeadScore'].max() * 100).round(1)
            
            logger.info(f"Loaded {len(geo_data):,} leads with geographical data")
            return geo_data
            
        except Exception as e:
            logger.error(f"Failed to load geographical data: {e}")
            raise
    
    def analyze_countries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead performance by country"""
        try:
            country_stats = df.groupby('CountryName').agg({
                'LeadId': 'count',
                'LeadScore': 'mean',
                'MeetingCount': lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0,
                'CallCount': 'mean',
                'LeadAge_Days': 'mean'
            }).reset_index()
            
            country_stats.columns = ['Country', 'LeadCount', 'AvgScore', 'ConversionRate', 'AvgCalls', 'AvgAge']
            
            total_leads = country_stats['LeadCount'].sum()
            country_stats['MarketShare'] = (country_stats['LeadCount'] / total_leads * 100).round(1)
            
            # Calculate market performance score
            country_stats['MarketScore'] = (
                (country_stats['MarketShare'] * 0.3) +
                (country_stats['AvgScore'] * 0.4) +
                (country_stats['ConversionRate'] * 0.3)
            ).round(1)
            
            country_stats = country_stats.sort_values('LeadCount', ascending=False).round(1)
            
            return {
                'total_countries': int(len(country_stats)),
                'total_leads': int(total_leads),
                'countries': country_stats.to_dict('records'),
                'top_3': country_stats.head(3)[['Country', 'LeadCount', 'MarketShare', 'MarketScore']].to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Country analysis failed: {e}")
            return {}
    
    def analyze_city_regions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lead performance by city/region"""
        try:
            if 'CityRegionName' not in df.columns:
                return {'available': False, 'message': 'City region data not available'}
            
            region_stats = df.groupby('CityRegionName').agg({
                'LeadId': 'count',
                'LeadScore': 'mean',
                'MeetingCount': 'sum',
                'CallCount': 'mean'
            }).reset_index()
            
            region_stats.columns = ['CityRegion', 'LeadCount', 'AvgScore', 'TotalMeetings', 'AvgCalls']
            
            total_leads = region_stats['LeadCount'].sum()
            region_stats['MarketShare'] = (region_stats['LeadCount'] / total_leads * 100).round(1)
            
            region_stats = region_stats.sort_values('LeadCount', ascending=False).head(20).round(1)
            
            return {
                'available': True,
                'total_regions': int(len(region_stats)),
                'regions': region_stats.to_dict('records'),
                'top_5': region_stats.head(5)[['CityRegion', 'LeadCount', 'MarketShare']].to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"City region analysis failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def generate_market_recommendations(self, df: pd.DataFrame, country_stats: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate market-specific recommendations"""
        recommendations = []
        
        try:
            # Best performing market
            if len(country_stats) > 0:
                best_market = country_stats.iloc[0]
                recommendations.append({
                    'type': 'best_market',
                    'priority': 'high',
                    'country': best_market['Country'],
                    'message': f"{best_market['Country']} is your top market with {int(best_market['LeadCount'])} leads ({best_market['MarketShare']}%)",
                    'metrics': {
                        'leads': int(best_market['LeadCount']),
                        'market_share': float(best_market['MarketShare']),
                        'market_score': float(best_market['MarketScore'])
                    }
                })
            
            # Low conversion markets
            for idx, row in country_stats.iterrows():
                if row['LeadCount'] > 10 and row['ConversionRate'] < 15:
                    recommendations.append({
                        'type': 'improvement_opportunity',
                        'priority': 'medium',
                        'country': row['Country'],
                        'message': f"{row['Country']} has {int(row['LeadCount'])} leads but only {row['ConversionRate']:.1f}% conversion rate",
                        'metrics': {
                            'leads': int(row['LeadCount']),
                            'conversion_rate': float(row['ConversionRate'])
                        }
                    })
            
            # Market concentration
            top3_share = country_stats.head(3)['MarketShare'].sum()
            if top3_share > 70:
                recommendations.append({
                    'type': 'concentration_warning',
                    'priority': 'medium',
                    'message': f"High market concentration: Top 3 countries account for {top3_share:.1f}% of leads",
                    'metrics': {
                        'top3_share': float(top3_share),
                        'diversification_needed': True
                    }
                })
            
            # Emerging markets
            for idx, row in country_stats.iterrows():
                if 5 < row['LeadCount'] < 20 and row['ConversionRate'] > 25:
                    recommendations.append({
                        'type': 'emerging_market',
                        'priority': 'high',
                        'country': row['Country'],
                        'message': f"{row['Country']} showing strong performance with {row['ConversionRate']:.1f}% conversion - consider expansion",
                        'metrics': {
                            'leads': int(row['LeadCount']),
                            'conversion_rate': float(row['ConversionRate'])
                        }
                    })
            
            return recommendations[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete geographical analysis"""
        try:
            # Connect
            if not self.connect():
                return {
                    'status': 'failed',
                    'error': 'Database connection failed',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Load data
            df = self.load_geographical_data()
            
            # Country analysis
            country_analysis = self.analyze_countries(df)
            
            # City region analysis
            city_region_analysis = self.analyze_city_regions(df)
            
            # Generate recommendations
            country_df = pd.DataFrame(country_analysis['countries'])
            recommendations = self.generate_market_recommendations(df, country_df)
            
            # Market concentration
            top3_share = country_df.head(3)['MarketShare'].sum() if len(country_df) >= 3 else 0
            
            # Prepare final response
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_leads': int(len(df)),
                    'total_countries': country_analysis['total_countries'],
                    'market_concentration': {
                        'top3_share': float(top3_share),
                        'status': 'High Concentration' if top3_share > 70 else 'Diversified'
                    }
                },
                'country_analysis': country_analysis,
                'city_region_analysis': city_region_analysis,
                'recommendations': recommendations
            }
            
            logger.info("Geographical analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            if self.engine:
                self.engine.dispose()
                