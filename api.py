from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Import agents
from agents.rag_agent import RAGAgent

# ============================
# FASTAPI APP SETUP
# ============================

app = FastAPI(
    title="Commodity AI System API",
    description="API for commodity market analysis with RAG-powered literature review",
    version="1.0.0"
)

# Global RAG agent instance
rag_agent: Optional[RAGAgent] = None

# ============================
# PYDANTIC MODELS
# ============================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    description: str = "Search query for literature review"

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class IntelligenceReportRequest(BaseModel):
    summarizer_output: Dict[str, Any]
    description: str = "Output from the summarizer agent"

class IntelligenceReportResponse(BaseModel):
    intelligence_report: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    pdf_count: int

# ============================
# ENDPOINTS
# ============================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG agent on startup"""
    global rag_agent
    try:
        print("[API] Initializing RAG agent...")
        rag_agent = RAGAgent(pdf_directory="data/literature_reviews")
        print("[API] RAG agent initialized successfully")
    except Exception as e:
        print(f"[API] Error initializing RAG agent: {e}")
        rag_agent = None

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    pdf_count = 0
    if rag_agent and rag_agent.documents:
        pdf_count = len(set(doc["source"] for doc in rag_agent.documents))
    
    return HealthResponse(
        status="healthy",
        rag_initialized=rag_agent is not None and rag_agent.index is not None,
        pdf_count=pdf_count
    )

@app.post("/query", response_model=QueryResponse)
async def search_literature(request: QueryRequest):
    """
    Search for relevant literature based on query
    
    Args:
        request: QueryRequest with query string and optional top_k parameter
        
    Returns:
        QueryResponse with search results
    """
    if rag_agent is None or rag_agent.index is None:
        raise HTTPException(
            status_code=503,
            detail="RAG agent not initialized. No PDFs indexed."
        )
    
    try:
        results = rag_agent.search_relevant_literature(
            query=request.query,
            top_k=request.top_k
        )
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during search: {str(e)}"
        )

@app.post("/intelligence-report", response_model=IntelligenceReportResponse)
async def generate_intelligence_report(request: IntelligenceReportRequest):
    """
    Generate intelligence report by cross-referencing summarizer output with literature
    
    Args:
        request: IntelligenceReportRequest with summarizer output
        
    Returns:
        IntelligenceReportResponse with intelligence findings
    """
    if rag_agent is None or rag_agent.index is None:
        raise HTTPException(
            status_code=503,
            detail="RAG agent not initialized. No PDFs indexed."
        )
    
    try:
        intelligence_report = rag_agent.generate_intelligence_report(
            request.summarizer_output
        )
        
        return IntelligenceReportResponse(
            intelligence_report=intelligence_report
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating intelligence report: {str(e)}"
        )

@app.get("/docs-info")
async def get_documentation_info():
    """Get information about available endpoints"""
    return {
        "endpoints": {
            "GET /": "Health check - verify API status",
            "POST /query": "Search literature by query",
            "POST /intelligence-report": "Generate intelligence report from summarizer output",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "example_queries": {
            "simple_query": {
                "endpoint": "/query",
                "body": {
                    "query": "crude oil price volatility impacts",
                    "top_k": 5
                }
            },
            "intelligence_report": {
                "endpoint": "/intelligence-report",
                "body": {
                    "summarizer_output": {
                        "oil": {
                            "sentiment": "bullish",
                            "drivers": ["geopolitical tensions", "supply constraints"],
                            "summary": "Oil prices rising due to supply concerns"
                        }
                    }
                }
            }
        }
    }

# ============================
# ERROR HANDLERS
# ============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

# ============================
# RUN
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
