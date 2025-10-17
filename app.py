"""
Lead Intelligence API - Complete Platform v3.0.0
Combines ML Lead Scoring + Geographical Analysis + Executive Dashboard
Production-Ready for Render Deployment
"""

import os
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any
import logging
from datetime import datetime

from ml_engine import AIMLModelsEngine
from geo_engine import GeographicalAnalysisEngine
from dashboard_engine import DashboardEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SETTINGS (Updated for Render Deployment)
# ============================================================================

class Settings(BaseSettings):
    # Database Configuration with Default Values (for Render)
    DATABASE_SERVER: str = Field(default="auto.resourceplus.app", env='DATABASE_SERVER')
    DATABASE_NAME: str = Field(default="ResourcePlus_Dev_Live", env='DATABASE_NAME')
    DATABASE_USERNAME: str = Field(default="ResourcePlus_BI", env='DATABASE_USERNAME')
    DATABASE_PASSWORD: str = Field(default="kN5yT#9mP@7qL2wX", env='DATABASE_PASSWORD')
    
    # API Configuration
    API_TITLE: str = Field(default="Lead Intelligence API", env='API_TITLE')
    API_VERSION: str = Field(default="3.0.0", env='API_VERSION')
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"

settings = Settings()

logger.info(f"🔧 Database Server: {settings.DATABASE_SERVER}")
logger.info(f"🔧 Database: {settings.DATABASE_NAME}")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="🎯 **Complete Lead Intelligence Platform** - ML Lead Scoring, Churn Prediction, Segmentation, Recommendations, Geographical Market Analysis + Executive Dashboard Analytics",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Lead Intelligence Team",
        "email": "support@leadintelligence.com"
    }
)

# CORS (Enable Cross-Origin Requests for Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info(f"🚀 {settings.API_TITLE} v{settings.API_VERSION} Starting...")
    logger.info(f"📊 Database: {settings.DATABASE_SERVER}/{settings.DATABASE_NAME}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info(f"🛑 {settings.API_TITLE} shutting down...")


# ============================================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    🏠 **Root Endpoint** - API Information
    """
    return {
        "service": "Lead Intelligence API",
        "version": settings.API_VERSION,
        "status": "active",
        "description": "Complete Lead Intelligence Platform with ML Models, Geographical Analysis + Executive Dashboard",
        "documentation": "/docs",
        "features": {
            "ml_models": [
                "Lead Scoring",
                "Churn Risk Prediction",
                "Lead Segmentation",
                "Smart Recommendations"
            ],
            "geographical_analysis": [
                "Country Performance",
                "Regional Analysis",
                "Market Recommendations",
                "Market Concentration"
            ],
            "dashboard": [
                "Performance KPIs (WTD, MTD, YTD)",
                "Trends Analysis (Last 6 Months)",
                "Lead Conversion Funnel",
                "Top Markets Analysis",
                "Executive Summary"
            ]
        },
        "endpoints": {
            # ML Endpoints
            "ml_complete": "POST /api/v1/score-all-leads",
            "ml_summary": "GET /api/v1/summary",
            "ml_top_leads": "GET /api/v1/top-leads/{limit}",
            "ml_at_risk": "GET /api/v1/at-risk-leads",
            "ml_recommendations": "GET /api/v1/recommendations",
            
            # Geo Endpoints
            "geo_complete": "POST /api/v1/geographical-analysis",
            "geo_countries": "GET /api/v1/countries",
            "geo_recommendations": "GET /api/v1/market-recommendations",
            
            # Dashboard Endpoints
            "dashboard_complete": "GET /api/v1/dashboard?date_filter=year",
            "dashboard_kpis": "GET /api/v1/dashboard/kpis?date_filter=year",
            "dashboard_trends": "GET /api/v1/dashboard/trends?date_filter=year",
            "dashboard_funnel": "GET /api/v1/dashboard/funnel?date_filter=year",
            "dashboard_markets": "GET /api/v1/dashboard/markets?date_filter=year",
            "executive_summary": "GET /api/v1/executive-summary?date_filter=year",
            
            # Health
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    ❤️ **Health Check** - Verify API and database connectivity
    """
    try:
        ml_engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        ml_status = ml_engine.connect_db()
        
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        geo_status = geo_engine.connect()
        
        dashboard_engine = DashboardEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        dashboard_status = dashboard_engine.connect()
        
        db_status = "connected" if (ml_status and geo_status and dashboard_status) else "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "ml_engine": "operational" if ml_status else "unavailable",
            "geo_engine": "operational" if geo_status else "unavailable",
            "dashboard_engine": "operational" if dashboard_status else "unavailable"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }


# ============================================================================
# ML MODELS ENDPOINTS
# ============================================================================

@app.post("/api/v1/score-all-leads", tags=["ML Models"])
def score_all_leads():
    """
    🎯 **Complete ML Analysis** - Score all active leads
    
    Runs all 4 ML models:
    - Lead Scoring (0-100)
    - Churn Risk Prediction
    - Lead Segmentation
    - Smart Recommendations
    
    **Response Time**: ~10-15 seconds for 1000+ leads
    
    **Returns**: Complete analysis with scores, segments, and recommendations
    """
    try:
        logger.info("Running complete ML analysis on all leads...")
        
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'ML analysis failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/summary", tags=["ML Models"])
def get_summary():
    """
    📊 **Quick Summary** - Get high-level metrics
    
    **Response Time**: ~10 seconds
    
    **Returns**: Total leads, avg score, priority distribution, at-risk count
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "summary": results.get('summary', {}),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/top-leads/{limit}", tags=["ML Models"])
def get_top_leads(limit: int = 10):
    """
    🏆 **Top Leads** - Get highest scoring leads
    
    **Parameters**:
    - limit: Number of leads to return (default: 10, max: 100)
    
    **Response Time**: ~10 seconds
    """
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        top_leads = results.get('top_leads', [])[:limit]
        
        return {
            "top_leads": top_leads,
            "count": len(top_leads),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Top leads retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/at-risk-leads", tags=["ML Models"])
def get_at_risk_leads():
    """
    ⚠️ **At-Risk Leads** - Get leads with high churn risk
    
    **Response Time**: ~10 seconds
    
    **Returns**: Leads that need immediate attention
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "at_risk_leads": results.get('at_risk_leads', []),
            "count": len(results.get('at_risk_leads', [])),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"At-risk leads retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/recommendations", tags=["ML Models"])
def get_recommendations():
    """
    💡 **Smart Recommendations** - Get AI-powered action items
    
    **Response Time**: ~10 seconds
    
    **Returns**: Prioritized recommendations for sales team
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "recommendations": results.get('recommendations', []),
            "count": len(results.get('recommendations', [])),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# GEOGRAPHICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/geographical-analysis", tags=["Geographical Analysis"])
def complete_geographical_analysis():
    """
    🌍 **Complete Geographical Analysis** - Market intelligence & performance
    
    Analyzes:
    - Lead distribution by country
    - Regional performance metrics
    - Market concentration
    - Country-specific recommendations
    
    **Response Time**: ~5-10 seconds
    
    **Returns**: Complete market analysis with countries, regions, and insights
    """
    try:
        logger.info("Running geographical analysis...")
        
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'Geographical analysis failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Geographical analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/countries", tags=["Geographical Analysis"])
def get_country_analysis():
    """
    🌍 **Country Analysis** - Get country-level metrics only
    
    **Response Time**: ~5 seconds
    
    **Returns**: Country performance metrics and rankings
    """
    try:
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error')
            )
        
        return {
            "country_analysis": results.get('country_analysis', {}),
            "timestamp": results['timestamp'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Country analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/market-recommendations", tags=["Geographical Analysis"])
def get_market_recommendations():
    """
    💡 **Market Recommendations** - Get geographical insights
    
    **Response Time**: ~5 seconds
    
    **Returns**: Market-specific strategic recommendations
    """
    try:
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error')
            )
        
        return {
            "recommendations": results.get('recommendations', []),
            "summary": results.get('summary', {}),
            "timestamp": results['timestamp'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Market recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/api/v1/dashboard", tags=["Dashboard"])
def get_complete_dashboard(
    date_filter: str = Query(default='year', description="Date filter: all_time, year, month, week, ytd")
):
    """
    📊 **Complete Executive Dashboard** - All metrics in one response
    
    **Parameters**:
    - date_filter: all_time, year, month, week, ytd (default: year)
    
    **Response Time**: ~3-5 seconds
    
    **Returns**:
    - Performance KPIs (WTD, MTD, YTD)
    - Performance Trends (Last 6 months)
    - Lead Conversion Funnel
    - Top 10 Markets
    - Executive Summary
    """
    try:
        dashboard_engine = DashboardEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = dashboard_engine.get_complete_dashboard(date_filter)
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'Dashboard generation failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/dashboard/kpis", tags=["Dashboard"])
def get_dashboard_kpis(date_filter: str = Query(default='year')):
    """
    📈 **KPIs Only** - Get performance metrics (WTD, MTD, YTD)
    
    **Response Time**: ~3 seconds
    """
    try:
        dashboard = get_complete_dashboard(date_filter)
        return {
            'status': 'success',
            'kpis': dashboard['kpis'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/trends", tags=["Dashboard"])
def get_dashboard_trends(date_filter: str = Query(default='year')):
    """
    📊 **Performance Trends** - Last 6 months metrics
    
    **Response Time**: ~3 seconds
    """
    try:
        dashboard = get_complete_dashboard(date_filter)
        return {
            'status': 'success',
            'trends': dashboard['trends'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/funnel", tags=["Dashboard"])
def get_dashboard_funnel(date_filter: str = Query(default='year')):
    """
    🎯 **Lead Funnel** - Conversion funnel by stage
    
    **Response Time**: ~3 seconds
    """
    try:
        dashboard = get_complete_dashboard(date_filter)
        return {
            'status': 'success',
            'funnel': dashboard['funnel'],
            'total_leads': dashboard.get('total_leads', 0),
            'total_converted': dashboard.get('total_converted', 0),
            'overall_conversion_rate': dashboard.get('overall_conversion_rate', 0.0),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/markets", tags=["Dashboard"])
def get_dashboard_markets(date_filter: str = Query(default='year')):
    """
    🌍 **Top Markets** - Top 10 countries by lead count
    
    **Response Time**: ~3 seconds
    """
    try:
        dashboard = get_complete_dashboard(date_filter)
        return {
            'status': 'success',
            'top_markets': dashboard['top_markets'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXECUTIVE SUMMARY ENDPOINT
# ============================================================================

@app.get("/api/v1/executive-summary", tags=["Executive Summary"])
def get_executive_summary(
    date_filter: str = Query(default='year', description="Date filter: all_time, year, month, week, ytd")
):
    """
    📋 **Executive Summary** - High-level overview for leadership
    
    **Parameters**:
    - date_filter: all_time, year, month, week, ytd (default: year)
    
    **Response Time**: ~2-3 seconds
    
    **Returns**:
    - Key business metrics (leads, conversion, revenue potential)
    - Performance indicators (WTD, MTD, YTD comparison)
    - Top performing markets
    - Critical alerts & recommendations
    - Business health score
    
    Perfect for executive dashboards and reports!
    """
    try:
        dashboard_engine = DashboardEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = dashboard_engine.get_executive_summary(date_filter)
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'Executive summary generation failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Executive summary generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# RUN SERVER (Updated for Render Deployment)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable from Render, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info"
    )
