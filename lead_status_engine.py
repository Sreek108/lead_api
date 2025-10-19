"""
Lead Status Analytics Engine for Lead Intelligence
Provides hot/cold lead classification, conversion rates, and status analytics
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any, List
import logging
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadStatusEngine:
    """Lead Status Analytics Engine - Hot/Cold Leads, Conversion, Status Distribution"""
    
    def __init__(self, server: str, database: str, username: str, password: str):
        """Initialize with database connection parameters"""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
    
    def connect(self) -> bool:
        """Establish database connection with connection pooling"""
        try:
            conn_string = f"mssql+pymssql://{self.username}:{quote_plus(self.password)}@{self.server}/{self.database}"
            self.engine = create_engine(
                conn_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10
            )
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("✅ Lead Status Engine connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def _categorize_leads(self, leads_df: pd.DataFrame, statuses_df: pd.DataFrame) -> pd.DataFrame:
        """Categorize leads as Hot, Cold, Won, or Other"""
        
        # Build status mapping
        status_col = 'StatusName_E' if 'StatusName_E' in statuses_df.columns else statuses_df.columns[1]
        name_map = dict(zip(statuses_df["LeadStatusId"].astype(int), statuses_df[status_col].astype(str)))
        
        L = leads_df.copy()
        L["Status"] = L["LeadStatusId"].map(name_map).fillna(L["LeadStatusId"].astype(str))
        L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
        L["age_days"] = (pd.Timestamp.now() - L["CreatedOn"]).dt.days.fillna(0).astype(int)
        
        # Categorization keywords
        hot_keywords = ['meeting scheduled', 'meeting confirmed']
        cold_keywords = ['follow-up needed', 'interested', 'uncontacted']
        won_keywords = ['won']
        
        # Categorize leads
        L['Category'] = 'Other'
        for idx, row in L.iterrows():
            status = str(row['Status']).lower()
            if any(k in status for k in won_keywords):
                L.at[idx, 'Category'] = 'Won'
            elif any(k in status for k in hot_keywords):
                L.at[idx, 'Category'] = 'Hot'
            elif any(k in status for k in cold_keywords):
                L.at[idx, 'Category'] = 'Cold'
        
        return L
    
    def get_complete_analytics(self) -> Dict[str, Any]:
        """
        Get complete lead status analytics
        Returns all sections: overview, metrics, distribution, trends, comparison, agent performance
        """
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            logger.info("🔄 Generating complete lead status analytics...")
            
            # Fetch required data
            leads = pd.read_sql("SELECT * FROM dbo.Lead WHERE IsActive = 1", self.engine)
            statuses = pd.read_sql("SELECT * FROM dbo.LeadStatus WHERE IsActive = 1", self.engine)
            meets = pd.read_sql("SELECT * FROM dbo.AgentMeetingAssignment", self.engine)
            
            if leads.empty:
                return {'status': 'failed', 'error': 'No leads data available'}
            
            # Categorize leads
            L = self._categorize_leads(leads, statuses)
            
            # Calculate key metrics
            total_leads = len(L)
            hot_leads = len(L[L['Category'] == 'Hot'])
            cold_leads = len(L[L['Category'] == 'Cold'])
            won_leads = len(L[L['Category'] == 'Won'])
            other_leads = len(L[L['Category'] == 'Other'])
            conversion_rate = round((won_leads / total_leads * 100), 2) if total_leads > 0 else 0
            avg_days_to_close = round(L[L['Category'] == 'Won']['age_days'].mean(), 1) if won_leads > 0 else 0
            
            # Status breakdown
            status_counts = L["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            top_3_statuses = status_counts.head(3).to_dict('records')
            
            # Status distribution with percentages
            status_distribution = []
            for idx, row in status_counts.iterrows():
                status_distribution.append({
                    'status': row['Status'],
                    'count': int(row['Count']),
                    'percentage': round((row['Count'] / total_leads * 100), 1)
                })
            
            # Status comparison matrix
            comparison = L.groupby('Status').agg(
                Total_Leads=('LeadId', 'count'),
                Avg_Age=('age_days', 'mean'),
                Won_Count=('Category', lambda x: (x == 'Won').sum())
            ).reset_index()
            
            comparison['Win_Rate'] = (comparison['Won_Count'] / comparison['Total_Leads'] * 100).round(1)
            comparison['Avg_Age'] = comparison['Avg_Age'].round(0)
            
            # Add meetings data if available
            if not meets.empty and 'LeadId' in meets.columns:
                try:
                    meeting_stats = meets.merge(L[['LeadId', 'Status']], on='LeadId', how='inner')
                    meeting_counts = meeting_stats.groupby('Status')['LeadId'].nunique().reset_index(name='Meetings')
                    comparison = comparison.merge(meeting_counts, on='Status', how='left')
                    comparison['Meetings'] = comparison['Meetings'].fillna(0).astype(int)
                except:
                    comparison['Meetings'] = 0
            else:
                comparison['Meetings'] = 0
            
            comparison['Health_Score'] = (
                (comparison['Win_Rate'] * 0.4) + 
                ((100 - comparison['Avg_Age'].clip(0, 100)) * 0.3) +
                ((comparison['Meetings'] / comparison['Total_Leads'].clip(1) * 100).clip(0, 100) * 0.3)
            ).round(0)
            
            status_comparison = comparison.sort_values('Total_Leads', ascending=False).head(10).to_dict('records')
            
            # Trends over time (last 6 months)
            L_trend = L.copy()
            L_trend["Month"] = L_trend["CreatedOn"].dt.to_period('M').astype(str)
            trend_data = L_trend.groupby(['Month', 'Status']).size().reset_index(name='Count')
            top_statuses = L["Status"].value_counts().head(6).index.tolist()
            trend_filtered = trend_data[trend_data["Status"].isin(top_statuses)]
            
            trends = []
            for month in sorted(trend_filtered['Month'].unique()):
                month_data = trend_filtered[trend_filtered['Month'] == month]
                trends.append({
                    'month': month,
                    'statuses': month_data[['Status', 'Count']].to_dict('records')
                })
            
            # Agent performance (if AssignedAgentId exists)
            agent_performance = []
            if "AssignedAgentId" in L.columns:
                L_with_agents = L[(L["AssignedAgentId"].notna()) & (L["AssignedAgentId"] != 0)].copy()
                
                if len(L_with_agents) > 0:
                    agent_stats = L_with_agents.groupby("AssignedAgentId").agg(
                        Total_Leads=("LeadId", "count"),
                        Won_Leads=("Category", lambda x: (x == 'Won').sum()),
                        Hot_Leads=("Category", lambda x: (x == 'Hot').sum()),
                        Cold_Leads=("Category", lambda x: (x == 'Cold').sum()),
                        Avg_Days=("age_days", "mean")
                    ).reset_index()
                    
                    agent_stats["Conversion_Rate"] = (agent_stats["Won_Leads"] / agent_stats["Total_Leads"] * 100).round(1)
                    agent_stats["Avg_Days"] = agent_stats["Avg_Days"].round(1)
                    agent_stats = agent_stats.sort_values("Conversion_Rate", ascending=False).head(10)
                    
                    agent_performance = agent_stats.to_dict('records')
            
            # Build complete response
            response = {
                'status': 'success',
                'overview': {
                    'total_leads': total_leads,
                    'hot_leads': hot_leads,
                    'cold_leads': cold_leads,
                    'won_leads': won_leads,
                    'other_leads': other_leads,
                    'conversion_rate': conversion_rate,
                    'avg_days_to_close': avg_days_to_close
                },
                'key_metrics': {
                    'total_leads': total_leads,
                    'won_deals': won_leads,
                    'win_rate': conversion_rate
                },
                'top_3_statuses': top_3_statuses,
                'status_distribution': status_distribution,
                'status_comparison_matrix': status_comparison,
                'trends': trends,
                'agent_performance': agent_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            self.engine.dispose()
            logger.info("✅ Lead status analytics generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ Lead status analytics generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_overview_only(self) -> Dict[str, Any]:
        """Get quick overview - Hot/Cold leads and conversion metrics only"""
        try:
            full_results = self.get_complete_analytics()
            
            if full_results.get('status') == 'failed':
                return full_results
            
            return {
                'status': 'success',
                'overview': full_results['overview'],
                'key_metrics': full_results['key_metrics'],
                'timestamp': full_results['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Overview generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_status_distribution(self) -> Dict[str, Any]:
        """Get status distribution with counts and percentages"""
        try:
            full_results = self.get_complete_analytics()
            
            if full_results.get('status') == 'failed':
                return full_results
            
            return {
                'status': 'success',
                'status_distribution': full_results['status_distribution'],
                'top_3_statuses': full_results['top_3_statuses'],
                'timestamp': full_results['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Status distribution generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_status_comparison(self) -> Dict[str, Any]:
        """Get detailed status comparison matrix with health scores"""
        try:
            full_results = self.get_complete_analytics()
            
            if full_results.get('status') == 'failed':
                return full_results
            
            return {
                'status': 'success',
                'status_comparison_matrix': full_results['status_comparison_matrix'],
                'timestamp': full_results['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Status comparison generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_trends(self) -> Dict[str, Any]:
        """Get status trends over time"""
        try:
            full_results = self.get_complete_analytics()
            
            if full_results.get('status') == 'failed':
                return full_results
            
            return {
                'status': 'success',
                'trends': full_results['trends'],
                'timestamp': full_results['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Trends generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance by status"""
        try:
            full_results = self.get_complete_analytics()
            
            if full_results.get('status') == 'failed':
                return full_results
            
            return {
                'status': 'success',
                'agent_performance': full_results['agent_performance'],
                'timestamp': full_results['timestamp']
            }
        except Exception as e:
            logger.error(f"❌ Agent performance generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
