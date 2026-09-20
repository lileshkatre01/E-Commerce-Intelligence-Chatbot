"""Backend package for E-commerce Review Intelligence & Analytics Chatbot."""

from backend.analytics import ReviewAnalytics, get_analytics
from backend.chatbot import (
    ReviewChatbot,
    ask_chatbot,
    ask_chatbot_with_sources,
    get_chatbot,
)
from backend.retriever import (
    ReviewMatch,
    ReviewRetriever,
    get_retriever,
    retrieve_reviews,
    retrieve_reviews_and_sources,
)

__all__ = [
    "ReviewChatbot",
    "ask_chatbot",
    "ask_chatbot_with_sources",
    "get_chatbot",
    "ReviewRetriever",
    "retrieve_reviews",
    "retrieve_reviews_and_sources",
    "get_retriever",
    "ReviewMatch",
    "ReviewAnalytics",
    "get_analytics",
]
