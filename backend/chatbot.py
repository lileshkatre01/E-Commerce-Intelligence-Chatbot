import os
from typing import Any, Dict, List, Optional
from google import genai

from backend.analytics import ReviewAnalytics, get_analytics
from backend.config import (
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SIMILARITY_THRESHOLD,
)
from backend.retriever import ReviewRetriever, extract_product_id_from_query


def is_catalog_or_general_query(query: str) -> bool:
    """Detect if query is asking about available products, catalog overview, or general capabilities."""
    q = query.lower().strip()
    catalog_keywords = [
        "product", "products", "avail", "catalog", "item", "items",
        "phone", "phones", "device", "devices", "list", "what do you have",
        "what can you", "overview", "summary", "sentiment", "complaint",
        "complaints", "store", "buy", "sell", "help", "hello", "hi",
        "who are you", "what is this", "csat", "ratings", "reviews"
    ]
    return any(k in q for k in catalog_keywords)



class ReviewChatbot:
    """Manages conversational review analysis combining:

    1. Qualitative RAG retrieval with Cosine Similarity & citations
    2. Quantitative E-commerce Analytics (Sentiment %, ratings, complaint topics)
    3. Gemini LLM prompt orchestration & strict grounding rules
    """

    def __init__(
        self,
        retriever: Optional[ReviewRetriever] = None,
        analytics: Optional[ReviewAnalytics] = None,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
    ):
        self.retriever = retriever or ReviewRetriever()
        self.analytics = analytics or get_analytics()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.chat_history: List[Dict[str, str]] = []
        self._client = None

    @property
    def client(self) -> Optional[genai.Client]:
        """Lazy-initialize Google GenAI client."""
        if self._client is None:
            key = self.api_key or os.getenv("GEMINI_API_KEY") or ""
            if not key or key.strip() in ("", "your_gemini_api_key_here"):
                return None
            try:
                self._client = genai.Client(api_key=key.strip())
            except Exception as init_err:
                print(f"[ReviewChatbot] GenAI client init error: {init_err}")
                return None
        return self._client

    def _format_history(self) -> str:
        """Format accumulated conversation history for the prompt."""
        if not self.chat_history:
            return "None"
        history_lines = []
        for entry in self.chat_history:
            history_lines.append(f"Question: {entry['question']}")
            history_lines.append(f"Answer: {entry['answer']}")
        return "\n".join(history_lines)

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        product_id: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Synthesize answers using both quantitative analytics and qualitative review context."""
        detected_product = product_id or extract_product_id_from_query(question)

        # 1. RAG Search (Qualitative Reviews)
        context, sources, matches = self.retriever.retrieve_context_and_sources(
            query=question,
            k=top_k,
            product_id=detected_product,
            min_rating=min_rating,
            max_rating=max_rating,
            similarity_threshold=similarity_threshold,
        )

        # 2. Analytics Engine (Quantitative Metrics)
        analytics_text = self.analytics.generate_analytics_context(product_id=detected_product)
        analytics_payload = (
            self.analytics.get_sentiment_summary(product_id=detected_product)
            if detected_product
            else self.analytics.get_executive_summary()
        )

        # GUARDRAIL: If no specific review matches pass threshold/filters:
        if not matches or not context.strip():
            # If asking about available products, catalog overview, or general capabilities:
            if is_catalog_or_general_query(question):
                p_stats = self.analytics.get_product_stats()
                catalog_lines = [
                    f"- Product {p['product_id']}: Rated {p['avg_rating']}/5.0 stars ({p['total_reviews']} reviews, {p['positive_pct']}% positive, status: {p['status']})"
                    for p in p_stats
                ]
                context = (
                    "Available Catalog Products Overview:\n"
                    + "\n".join(catalog_lines)
                    + "\n\nTotal catalog size: 10 products (P101 through P110), 50 customer reviews analyzed."
                )
                sources = [p["product_id"] for p in p_stats]
            else:
                filter_note = f" for product {detected_product}" if detected_product else ""
                rating_note = ""
                if min_rating or max_rating:
                    rating_note = f" with rating criteria (min: {min_rating}, max: {max_rating})"

                refusal_message = (
                    f"Based on the customer reviews available, there is no relevant feedback{filter_note}{rating_note} "
                    "to answer this question. Please ensure your query relates to customer reviews of available products (P101 through P110)."
                )
                return {
                    "question": question,
                    "answer": refusal_message,
                    "sources": [],
                    "source_details": [],
                    "context": "",
                    "product_filter_applied": detected_product,
                    "analytics": analytics_payload,
                }

        history_str = self._format_history()

        prompt = f"""You are an E-commerce Customer Review Intelligence Assistant.

STRICT GROUNDING RULES:
1. Answer using BOTH the quantitative analytics metrics and the qualitative customer reviews / catalog context below.
2. When citing facts or user sentiment, incorporate exact statistics (e.g. percentages, average rating).
3. If citing specific reviews, reference them inline using bracketed citation tags like [Review 7] or [Review 12]. If describing products from the catalog overview, refer to their product IDs (e.g. P101, P102).
4. DO NOT assume, extrapolate, or bring in external knowledge not present in the reviews or catalog.
5. If the provided data does not contain sufficient detail to fully answer a question, state:
   "Based on the customer reviews provided, there is no mention of [specific topic]."
6. Keep your answer conversational, objective, and analytical.

Conversation history:
{history_str}

Quantitative E-Commerce Analytics:
{analytics_text}

Qualitative Customer Reviews / Catalog Context:
{context}

User Question:
{question}
"""

        answer = ""
        if self.client is not None:
            models_to_try = [self.model] + [m for m in ["gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"] if m != self.model]
            for m in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    if response.text:
                        answer = response.text
                        break
                except Exception as gen_err:
                    print(f"[ReviewChatbot] Model {m} notice: {gen_err}")

        # Fallback to grounded rule-based synthesis if Gemini is unavailable or not configured
        if not answer:
            lines = [
                "**Customer Review Grounded Summary:**",
            ]
            if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
                lines.append("*(Note: Running in local grounded mode. To activate Gemini 3.6 Flash, insert your key in `.env`)*\n")

            if not matches and is_catalog_or_general_query(question):
                p_stats = self.analytics.get_product_stats()
                lines.append("We monitor customer feedback across **10 products** in our catalog:\n")
                for p in p_stats:
                    lines.append(f"• **{p['product_id']}**: {p['avg_rating']}/5.0 stars ({p['positive_pct']}% positive) — {p['status']}")
            else:
                if detected_product:
                    summary = self.analytics.get_sentiment_summary(detected_product)
                    lines.append(
                        f"• **Product {detected_product}:** Rated **{summary.get('avg_rating', 0)}/5 stars** on average "
                        f"({summary.get('positive_pct', 0)}% positive, {summary.get('negative_pct', 0)}% critical complaints)."
                    )

                if matches:
                    lines.append("\n**Key Grounded Findings from Retrieved Customer Reviews:**")
                    for m in matches[:4]:
                        lines.append(f"• [{m.source}] ({m.rating}/5 stars rating, Cosine relevance {m.similarity:.2f}): \"{m.review_text}\"")

            answer = "\n".join(lines)

        # Update conversation history
        self.chat_history.append({"question": question, "answer": answer})

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "source_details": [m.to_dict() for m in matches],
            "context": context,
            "product_filter_applied": detected_product,
            "analytics": analytics_payload,
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return current conversation history."""
        return list(self.chat_history)


# Global singleton instance and functional interface
_default_chatbot: Optional[ReviewChatbot] = None


def get_chatbot() -> ReviewChatbot:
    """Get or initialize default ReviewChatbot instance."""
    global _default_chatbot
    if _default_chatbot is None:
        _default_chatbot = ReviewChatbot()
    return _default_chatbot


def ask_chatbot(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    product_id: Optional[str] = None,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
) -> str:
    """Send question through the hybrid RAG + Analytics pipeline and return the answer string."""
    bot = get_chatbot()
    result = bot.ask(
        question=question,
        top_k=top_k,
        product_id=product_id,
        min_rating=min_rating,
        max_rating=max_rating,
        similarity_threshold=similarity_threshold,
    )
    return result["answer"]


def ask_chatbot_with_sources(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    product_id: Optional[str] = None,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Send question through the pipeline and return answer, sources, details, and analytics."""
    bot = get_chatbot()
    return bot.ask(
        question=question,
        top_k=top_k,
        product_id=product_id,
        min_rating=min_rating,
        max_rating=max_rating,
        similarity_threshold=similarity_threshold,
    )
