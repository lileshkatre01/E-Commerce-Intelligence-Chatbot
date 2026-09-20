"""Root-level entry for chatbot module.

Allows direct imports:
    from chatbot import ask_chatbot, ask_chatbot_with_sources, get_analytics
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    "ask_chatbot",
    "ask_chatbot_with_sources",
    "ReviewChatbot",
    "get_chatbot",
    "retrieve_reviews",
    "retrieve_reviews_and_sources",
    "ReviewRetriever",
    "get_retriever",
    "ReviewMatch",
    "ReviewAnalytics",
    "get_analytics",
]

if __name__ == "__main__":
    test_question = "What is the overall sentiment of customer reviews?"
    print(f"Question: {test_question}\n")
    answer = ask_chatbot(test_question)
    print("Answer:\n", answer)
