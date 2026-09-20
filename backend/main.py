import argparse
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.analytics import ReviewAnalytics
from backend.chatbot import ReviewChatbot
from backend.config import DEFAULT_TOP_K, GEMINI_MODEL, SIMILARITY_THRESHOLD
from backend.retriever import ReviewRetriever

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Initialize FastAPI application
app = FastAPI(
    title="E-commerce Customer Review Intelligence & Analytics API",
    description=(
        "Hybrid AI Chatbot API combining RAG vector search (FAISS + Gemini) "
        "and Data Science Analytics (Pandas + NLP sentiment/complaint mining)."
    ),
    version="3.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton instances
retriever = ReviewRetriever()
analytics = ReviewAnalytics()
chatbot = ReviewChatbot(retriever=retriever, analytics=analytics)


class ChatRequest(BaseModel):
    message: Optional[str] = Field(
        None,
        description="The customer inquiry or question.",
        examples=["What problems are customers facing with P102?"],
    )
    question: Optional[str] = Field(
        None,
        description="Alternative field for the question.",
    )
    product_id: Optional[str] = Field(
        None,
        description="Optional product ID filter (e.g. 'P101', 'P102'). Auto-detected from query if omitted.",
        examples=["P102"],
    )
    min_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Optional minimum star rating (1-5).",
        examples=[1],
    )
    max_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Optional maximum star rating (1-5). E.g. max_rating=2 to analyze only complaints.",
        examples=[2],
    )
    similarity_threshold: Optional[float] = Field(
        SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine similarity cutoff score (default: 0.35). Reviews below this threshold are ignored.",
        examples=[0.35],
    )
    top_k: Optional[int] = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=20,
        description="Number of top relevant reviews to retrieve.",
        examples=[5],
    )

    @property
    def query(self) -> str:
        return (self.message or self.question or "").strip()


class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Grounded AI response generated from customer reviews with inline citations.",
    )
    sources: List[str] = Field(
        ...,
        description="List of referenced review source labels.",
        examples=[["Review 6 (P102, 2 stars)", "Review 8 (P102, 1 stars)"]],
    )
    source_details: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Detailed review objects including product ID, rating, review text, and similarity score.",
    )
    product_filter_applied: Optional[str] = Field(
        None,
        description="The product ID filter applied (explicit or auto-detected).",
    )
    analytics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured quantitative e-commerce metrics relevant to the query.",
    )


# ---------------------------------------------------------
# Core API Endpoints
# ---------------------------------------------------------

@app.get("/api")
def root():
    return {
        "project": "E-commerce Customer Review Intelligence & Analytics API",
        "status": "online",
        "model": GEMINI_MODEL,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "docs_url": "/docs",
    }



@app.get("/health")
def health_check():
    try:
        doc_count = len(retriever.documents) if retriever.documents else 0
        df_count = len(analytics.get_df())
        return {
            "status": "healthy",
            "indexed_documents": doc_count,
            "analytics_records": df_count,
            "similarity_metric": "Cosine Similarity (IndexFlatIP)",
            "model": GEMINI_MODEL,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Hybrid chat endpoint synthesizing RAG vector search, live analytics metrics,

    and Gemini generation with inline citations.
    """
    user_query = request.query
    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Request must contain a non-empty 'message' or 'question' field.",
        )

    try:
        result = chatbot.ask(
            question=user_query,
            top_k=request.top_k or DEFAULT_TOP_K,
            product_id=request.product_id,
            min_rating=request.min_rating,
            max_rating=request.max_rating,
            similarity_threshold=request.similarity_threshold,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            source_details=result.get("source_details", []),
            product_filter_applied=result.get("product_filter_applied"),
            analytics=result.get("analytics"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_endpoint():
    """Reset conversational memory."""
    chatbot.clear_history()
    return {"message": "Chat history reset successfully."}


@app.get("/history")
def get_history_endpoint():
    """Retrieve full conversation history."""
    return {"history": chatbot.get_history()}


@app.post("/reindex")
def reindex_endpoint():
    """Force rebuilding and re-caching of the FAISS vector index."""
    try:
        retriever.build_or_load_index(force_rebuild=True)
        analytics.reload()
        return {
            "message": "Index and analytics reloaded successfully.",
            "documents_count": len(retriever.documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {e}")


# ---------------------------------------------------------
# Dedicated E-Commerce Analytics Endpoints
# ---------------------------------------------------------

@app.get("/analytics/summary")
def get_analytics_summary():
    """Executive KPI summary: total reviews, average rating, sentiment split,

    and top/bottom performing products.
    """
    try:
        return analytics.get_executive_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")


@app.get("/analytics/products")
def get_product_analytics(product_id: Optional[str] = Query(None, description="Optional product filter")):
    """Product-level analytics breakdown: average ratings, sentiment breakdown,

    and star distributions.
    """
    try:
        stats = analytics.get_product_stats()
        if product_id:
            pid = product_id.upper().strip()
            filtered = [p for p in stats if p["product_id"] == pid]
            if not filtered:
                raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
            return filtered[0]
        return {"total_products": len(stats), "products": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")


@app.get("/analytics/complaints")
def get_complaint_analytics():
    """Mining of common complaint topics across all customer reviews,

    their frequency, and affected products.
    """
    try:
        return analytics.get_complaint_topics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")


# ---------------------------------------------------------
# Mount Enterprise UI Frontend
# ---------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



def run_cli():
    """Run an interactive CLI session directly in the terminal."""
    print("=" * 60)
    print("  E-commerce Review Intelligence & Analytics (CLI)")
    print("=" * 60)
    print("Type your question (e.g. 'What is the overall sentiment?')")
    print("Commands: 'exit' or 'quit' | 'reset' to clear history\n")

    print("Initializing vector index & analytics...")
    retriever.build_or_load_index()
    print("Ready!\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if user_input.lower() == "reset":
                chatbot.clear_history()
                print("[Chat history reset.]\n")
                continue

            print("\nSearching reviews & computing analytics...")
            result = chatbot.ask(user_input)
            print(f"\nAssistant:\n{result['answer']}")
            if result.get("sources"):
                print(f"\nSources Cited: {', '.join(result['sources'])}")
            if result.get("product_filter_applied"):
                print(f"[Product Filter Auto-Applied: {result['product_filter_applied']}]")
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review Intelligence & Analytics")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI mode instead of starting API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000)",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        import uvicorn

        print(f"Starting FastAPI server on http://{args.host}:{args.port} ...")
        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=True)
