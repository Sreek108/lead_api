"""
AI/ML Models Engine for Lead Intelligence
Provides lead scoring, churn prediction, segmentation, and recommendations
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any, List
import logging
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIMLModelsEngine:
    """
    AI/ML Models Engine - Complete Lead Intelligence Platform
    Includes: Lead Scoring, Churn Risk, Segmentation, Recommendations
    """
    
    def __init__(self, server: str, database: str, username: str, password: str):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        
    def connect_db(self) -> bool:
        """Establish database connection with connection pooling"""
        try:
            conn_string = f"mssql+pymssql://{self.username}:{quote_plus(self.password)}@{self.server}/{self.database}?charset=utf8"
            
            self.engine = create_engine(
                conn_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10,
                connect_args={
                    'timeout': 30,
                    'login_timeout': 30
                },
                echo=False
            )
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("ML Engine: Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"ML Engine: Connection failed: {e}")
            return False
    
    def load_lead_data(self) -> pd.DataFrame:
        """Load all lead data with features (including inactive leads)"""
        try:
            logger.info("ML Engine: Loading lead data...")
            
            # Load ALL leads (including inactive) - UPDATED
            leads_query = """
            SELECT LeadId, LeadCode, LeadStatusId, CreatedOn, CountryId, CityRegionId, IsActive
            FROM dbo.Lead
            """
            leads = pd.read_sql(leads_query, self.engine)
            
            # Load meetings
            meetings = pd.read_sql("SELECT LeadId, StartDateTime FROM dbo.AgentMeetingAssignment", self.engine)
            meeting_counts = meetings.groupby('LeadId').size().reset_index(name='MeetingCount')
            
            # Load calls
            calls = pd.read_sql("SELECT LeadId, CallDateTime FROM dbo.LeadCallRecord", self.engine)
            call_counts = calls.groupby('LeadId').size().reset_index(name='CallCount')
            
            # Merge data
            leads = leads.merge(meeting_counts, on='LeadId', how='left')
            leads = leads.merge(call_counts, on='LeadId', how='left')
            leads['MeetingCount'] = leads['MeetingCount'].fillna(0)
            leads['CallCount'] = leads['CallCount'].fillna(0)
            
            # Calculate lead age
            leads['CreatedOn'] = pd.to_datetime(leads['CreatedOn'])
            leads['LeadAge_Days'] = (datetime.now() - leads['CreatedOn']).dt.days
            
            # Log with breakdown - UPDATED
            active_count = leads[leads['IsActive'] == 1].shape[0]
            inactive_count = leads[leads['IsActive'] == 0].shape[0]
            logger.info(f"ML Engine: Loaded {len(leads)} leads ({active_count} active, {inactive_count} inactive)")
            
            return leads
            
        except Exception as e:
            logger.error(f"ML Engine: Data loading failed: {e}")
            raise
    
    def calculate_lead_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lead score (0-100) based on multiple factors"""
        logger.info("ML Engine: Calculating lead scores...")
        
        # Initialize score
        df['LeadScore'] = 50.0
        
        # Factor 1: Meeting count (0-30 points)
        df['LeadScore'] += df['MeetingCount'].clip(0, 5) * 6
        
        # Factor 2: Call engagement (0-20 points)
        df['LeadScore'] += df['CallCount'].clip(0, 10) * 2
        
        # Factor 3: Lead age (fresher = better) (-10 to +10 points)
        df['LeadScore'] -= (df['LeadAge_Days'] / 30).clip(-10, 10)
        
        # Factor 4: Status bonus (high-value statuses get boost)
        high_value_statuses = [4, 5, 6]  # Meeting Scheduled, Negotiation, etc.
        df['LeadScore'] += df['LeadStatusId'].isin(high_value_statuses) * 10
        
        # Normalize to 0-100
        df['LeadScore'] = df['LeadScore'].clip(0, 100).round(1)
        
        logger.info(f"ML Engine: Lead scores calculated - Avg: {df['LeadScore'].mean():.1f}")
        return df
    
    def predict_churn_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn risk percentage (0-100%)"""
        logger.info("ML Engine: Calculating churn risk...")
        
        # Initialize risk
        df['ChurnRisk'] = 30.0
        
        # Risk Factor 1: No recent activity
        df['ChurnRisk'] += (df['LeadAge_Days'] > 30) * 20
        df['ChurnRisk'] += (df['LeadAge_Days'] > 60) * 20
        
        # Risk Factor 2: Low engagement
        df['ChurnRisk'] += (df['CallCount'] == 0) * 15
        df['ChurnRisk'] += (df['MeetingCount'] == 0) * 15
        
        # Risk Factor 3: Stuck in early stages
        early_stages = [1, 2, 3]
        df['ChurnRisk'] += (df['LeadStatusId'].isin(early_stages) & (df['LeadAge_Days'] > 14)) * 10
        
        # Protection: Active engagement reduces risk
        df['ChurnRisk'] -= df['MeetingCount'].clip(0, 3) * 10
        df['ChurnRisk'] -= df['CallCount'].clip(0, 5) * 2
        
        # Normalize
        df['ChurnRisk'] = df['ChurnRisk'].clip(0, 100).round(1)
        
        # Flag at-risk leads (>60% risk)
        df['IsAtRisk'] = df['ChurnRisk'] > 60
        
        logger.info(f"ML Engine: Churn risk calculated - {df['IsAtRisk'].sum()} leads at risk")
        return df
    
    def segment_leads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Segment leads into strategic categories"""
        logger.info("ML Engine: Segmenting leads...")
        
        def assign_segment(row):
            score = row['LeadScore']
            meetings = row['MeetingCount']
            calls = row['CallCount']
            age = row['LeadAge_Days']
            
            if meetings >= 2 or score >= 75:
                return 'Hot Prospects'
            elif meetings == 1 or (calls >= 3 and score >= 60):
                return 'Engaged Nurturers'
            elif calls >= 1 and score >= 40:
                return 'Long-term Opportunities'
            elif age < 7:
                return 'New Leads'
            else:
                return 'Cold Leads'
        
        df['Segment'] = df.apply(assign_segment, axis=1)
        
        segment_counts = df['Segment'].value_counts().to_dict()
        logger.info(f"ML Engine: Leads segmented into {len(segment_counts)} categories")
        
        return df
    
    def assign_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign priority based on lead score"""
        logger.info("ML Engine: Assigning priorities...")
        
        df['Priority'] = 'Medium'
        df.loc[df['LeadScore'] >= 70, 'Priority'] = 'High'
        df.loc[df['LeadScore'] < 40, 'Priority'] = 'Low'
        
        return df
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        logger.info("ML Engine: Generating recommendations...")
        
        recommendations = []
        
        # High-risk leads
        at_risk_count = df['IsAtRisk'].sum()
        if at_risk_count > 0:
            recommendations.append({
                'priority': 'urgent',
                'category': 'Churn Prevention',
                'title': f'{at_risk_count} Leads at High Churn Risk',
                'description': f'Immediate action required for {at_risk_count} leads with >60% churn risk',
                'action': 'Schedule follow-up calls and re-engagement campaigns',
                'impact': 'high',
                'affected_leads': int(at_risk_count)
            })
        
        # Cold leads
        cold_leads = len(df[df['Segment'] == 'Cold Leads'])
        if cold_leads > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'Re-engagement',
                'title': f'{cold_leads} Cold Leads Need Attention',
                'description': f'{cold_leads} leads showing low engagement',
                'action': 'Launch targeted email campaign and special offers',
                'impact': 'medium',
                'affected_leads': int(cold_leads)
            })
        
        # Hot prospects
        hot_leads = len(df[df['Segment'] == 'Hot Prospects'])
        if hot_leads > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'Conversion Opportunity',
                'title': f'{hot_leads} Hot Prospects Ready to Convert',
                'description': f'{hot_leads} highly engaged leads showing strong buying signals',
                'action': 'Prioritize for immediate follow-up and closing',
                'impact': 'high',
                'affected_leads': int(hot_leads)
            })
        
        # No meetings
        no_meetings = len(df[df['MeetingCount'] == 0])
        if no_meetings > df.shape[0] * 0.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'Meeting Conversion',
                'title': f'{no_meetings} Leads Without Meetings',
                'description': f'{(no_meetings/len(df)*100):.1f}% of leads have no scheduled meetings',
                'action': 'Increase meeting booking efforts and optimize scheduling',
                'impact': 'medium',
                'affected_leads': int(no_meetings)
            })
        
        logger.info(f"ML Engine: Generated {len(recommendations)} recommendations")
        return recommendations
    
    def run_all_models(self) -> Dict[str, Any]:
        """Run all ML models and return comprehensive analysis"""
        try:
            logger.info("="*60)
            logger.info("ML Engine: Starting complete analysis...")
            logger.info("="*60)
            
            if not self.connect_db():
                return {
                    'status': 'failed',
                    'error': 'Database connection failed',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Load data
            df = self.load_lead_data()
            
            if df.empty:
                return {
                    'status': 'failed',
                    'error': 'No leads found in database',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Run all models
            df = self.calculate_lead_score(df)
            df = self.predict_churn_risk(df)
            df = self.segment_leads(df)
            df = self.assign_priority(df)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(df)
            
            # Get top leads
            top_leads = df.nlargest(10, 'LeadScore')[['LeadCode', 'LeadScore', 'Priority', 'Segment', 'MeetingCount', 'CallCount']].to_dict('records')
            
            # Get at-risk leads
            at_risk_leads = df[df['IsAtRisk'] == True].nlargest(10, 'ChurnRisk')[['LeadCode', 'ChurnRisk', 'LeadAge_Days', 'CallCount', 'MeetingCount']].to_dict('records')
            
            # Segment distribution
            segments = df['Segment'].value_counts().to_dict()
            
            # Prepare result - UPDATED WITH ACTIVE/INACTIVE COUNTS
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_leads': int(len(df)),  # All leads (20)
                    'active_leads': int(df[df['IsActive'] == 1].shape[0]),  # Active (19)
                    'inactive_leads': int(df[df['IsActive'] == 0].shape[0]),  # Inactive (1)
                    'average_lead_score': float(df['LeadScore'].mean().round(1)),
                    'high_priority_leads': int((df['Priority'] == 'High').sum()),
                    'medium_priority_leads': int((df['Priority'] == 'Medium').sum()),
                    'low_priority_leads': int((df['Priority'] == 'Low').sum()),
                    'at_risk_leads': int(df['IsAtRisk'].sum()),
                    'average_churn_risk': float(df['ChurnRisk'].mean().round(1))
                },
                'segments': {str(k): int(v) for k, v in segments.items()},
                'top_leads': top_leads,
                'at_risk_leads': at_risk_leads,
                'recommendations': recommendations
            }
            
            logger.info("="*60)
            logger.info("ML Engine: Analysis completed successfully!")
            logger.info("="*60)
            
            return result
            
        except Exception as e:
            logger.error(f"ML Engine: Analysis failed: {e}")
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
