import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from backend.config import (
    CANDIDATE_POOL_MULTIPLIER,
    DATASET_URL,
    DOCUMENTS_PICKLE_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    FAISS_INDEX_PATH,
    REVIEWS_CSV_PATH,
    SIMILARITY_THRESHOLD,
)


class ReviewMatch:
    """Represents a retrieved review with metadata and similarity score."""

    def __init__(
        self,
        review_id: str,
        product_id: str,
        rating: int,
        review_text: str,
        similarity: float,
        source: str,
    ):
        self.review_id = str(review_id)
        self.product_id = str(product_id)
        self.rating = int(rating) if str(rating).isdigit() else rating
        self.review_text = review_text
        self.similarity = round(float(similarity), 4)
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "product_id": self.product_id,
            "rating": self.rating,
            "review_text": self.review_text,
            "similarity": self.similarity,
            "source": self.source,
        }

    def format_for_llm(self) -> str:
        return (
            f"[{self.source}]\n"
            f"Product: {self.product_id}\n"
            f"Rating : {self.rating}/5 stars\n"
            f"Review : {self.review_text}"
        )


def extract_product_id_from_query(query: str) -> Optional[str]:
    """Auto-detect product ID mentioned in text (e.g. 'P101', 'p102')."""
    match = re.search(r"\b(P\d{3})\b", query, re.IGNORECASE)
    return match.group(1).upper() if match else None


class ReviewRetriever:
    """Handles loading reviews, normalized embeddings for Cosine Similarity,

    FAISS IndexFlatIP, metadata filtering, and threshold validation.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        csv_path: Path = REVIEWS_CSV_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        docs_path: Path = DOCUMENTS_PICKLE_PATH,
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.index_path = Path(index_path)
        self.docs_path = Path(docs_path)

        self._model = None
        self._index = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    @property
    def model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def index(self):
        """Lazy access to FAISS index."""
        if self._index is None:
            self.build_or_load_index()
        return self._index

    def ensure_data_exists(self) -> None:
        """Download reviews.csv if missing."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading reviews dataset from {DATASET_URL}...")
            response = requests.get(DATASET_URL, timeout=30)
            response.raise_for_status()
            self.csv_path.write_text(response.text, encoding="utf-8")
            print(f"Saved dataset to {self.csv_path}")

    def load_documents_from_csv(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load review documents and metadata from CSV."""
        self.ensure_data_exists()
        df = pd.read_csv(self.csv_path)
        docs = []
        meta = []
        for _, row in df.iterrows():
            review_id = str(row.get("review_id", "")).strip()
            product_id = str(row.get("product_id", "N/A")).strip().upper()
            rating = int(row.get("rating", 0))
            review_text = str(row.get("review_text", "")).strip()

            source_label = f"Review {review_id}" if review_id else f"Review for {product_id}"

            text = (
                f"Source: {source_label}\n"
                f"Product: {product_id}\n"
                f"Rating : {rating}/5 stars\n"
                f"Review : {review_text}"
            )
            docs.append(text.strip())
            meta.append(
                {
                    "source": source_label,
                    "review_id": review_id,
                    "product_id": product_id,
                    "rating": rating,
                    "review_text": review_text,
                }
            )
        return docs, meta

    def build_or_load_index(self, force_rebuild: bool = False):
        """Build or load FAISS IndexFlatIP with L2-normalized embeddings for Cosine Similarity."""
        import faiss

        if not force_rebuild and self.index_path.exists() and self.docs_path.exists():
            print(f"Loading existing FAISS index from {self.index_path}...")
            self._index = faiss.read_index(str(self.index_path))
            with open(self.docs_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", [])
                else:
                    self.documents = data
                    self.metadata = []
            print(f"Loaded {len(self.documents)} indexed documents.")
            return self._index

        print("Building new FAISS Cosine Similarity index from reviews...")
        self.documents, self.metadata = self.load_documents_from_csv()
        if not self.documents:
            raise ValueError("No review documents found to build index.")

        print(f"Encoding {len(self.documents)} documents with {self.model_name}...")
        embeddings = self.model.encode(self.documents)
        embeddings = np.array(embeddings, dtype=np.float32)

        # L2-normalize vectors so Inner Product (IndexFlatIP) equals Cosine Similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        # Persist index and metadata
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.docs_path, "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)

        print(f"Index successfully built and persisted at {self.index_path}")
        return self._index

    def retrieve_matches(
        self,
        query: str,
        k: int = 5,
        product_id: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> List[ReviewMatch]:
        """Retrieve top k reviews matching query, filters, and similarity threshold."""
        import faiss

        if self._index is None:
            self.build_or_load_index()

        # Auto-detect product ID from query if not explicitly passed
        target_product = product_id.upper().strip() if product_id else extract_product_id_from_query(query)

        # Encode & L2-normalize query vector for Cosine Similarity
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # If metadata filters are requested, search the entire collection to avoid starvation
        has_filters = bool(target_product or min_rating is not None or max_rating is not None)
        if has_filters:
            candidate_count = len(self.documents)
        else:
            candidate_count = min(len(self.documents), max(k * CANDIDATE_POOL_MULTIPLIER, 20))

        similarities, indices = self._index.search(query_vec, k=candidate_count)

        matches: List[ReviewMatch] = []
        thresh = similarity_threshold if similarity_threshold is not None else -1.0

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            # Cosine similarity threshold guard
            if sim < thresh:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            item_product = str(meta.get("product_id", "")).upper()
            item_rating = meta.get("rating", 0)

            # Metadata Filter: Product ID
            if target_product and item_product != target_product:
                continue

            # Metadata Filter: Rating range
            if min_rating is not None and item_rating < min_rating:
                continue
            if max_rating is not None and item_rating > max_rating:
                continue

            match = ReviewMatch(
                review_id=meta.get("review_id", str(idx + 1)),
                product_id=meta.get("product_id", "Unknown"),
                rating=item_rating,
                review_text=meta.get("review_text", self.documents[idx]),
                similarity=sim,
                source=meta.get("source", f"Review {idx + 1}"),
            )
            matches.append(match)

            if len(matches) >= k:
                break

        return matches

    def retrieve_context_and_sources(
        self,
        query: str,
        k: int = 5,
        product_id: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> Tuple[str, List[str], List[ReviewMatch]]:
        """Returns: (formatted_context, source_strings, list_of_matches)."""
        matches = self.retrieve_matches(
            query=query,
            k=k,
            product_id=product_id,
            min_rating=min_rating,
            max_rating=max_rating,
            similarity_threshold=similarity_threshold,
        )

        if not matches:
            return "", [], []

        formatted_docs = [m.format_for_llm() for m in matches]
        context = "\n\n".join(formatted_docs)
        sources = [f"{m.source} ({m.product_id}, {m.rating} stars)" for m in matches]
        return context, sources, matches

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Simple helper for concatenated review context."""
        context, _, _ = self.retrieve_context_and_sources(query, k=k)
        return context


# Global singleton instance and functional interface
_default_retriever: Optional[ReviewRetriever] = None


def get_retriever() -> ReviewRetriever:
    """Get or initialize the default ReviewRetriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = ReviewRetriever()
    return _default_retriever


def retrieve_reviews(query: str, k: int = 5) -> str:
    """Convenience function: Query the FAISS vector index for top k relevant reviews."""
    return get_retriever().retrieve_context(query, k=k)


def retrieve_reviews_and_sources(
    query: str,
    k: int = 5,
    product_id: Optional[str] = None,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
) -> Tuple[str, List[str]]:
    """Retrieve both formatted context and source labels with filtering."""
    context, sources, _ = get_retriever().retrieve_context_and_sources(
        query=query,
        k=k,
        product_id=product_id,
        min_rating=min_rating,
        max_rating=max_rating,
        similarity_threshold=similarity_threshold,
    )
    return context, sources
