"""Root-level entry for retriever module.

Allows direct imports:
    from retriever import retrieve_reviews, retrieve_reviews_and_sources, ReviewRetriever
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.retriever import (
    ReviewRetriever,
    get_retriever,
    retrieve_reviews,
    retrieve_reviews_and_sources,
)

__all__ = [
    "ReviewRetriever",
    "get_retriever",
    "retrieve_reviews",
    "retrieve_reviews_and_sources",
]

if __name__ == "__main__":
    test_query = "poor battery life"
    print(f"Query: {test_query}\n")
    reviews, sources = retrieve_reviews_and_sources(test_query, k=3)
    print("Retrieved Sources:", sources)
    print("Retrieved Reviews:\n", reviews)
