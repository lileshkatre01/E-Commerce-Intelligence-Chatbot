import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.config import (
    DATA_DIR,
    DOCUMENTS_PICKLE_PATH,
    FAISS_INDEX_PATH,
    REVIEWS_CSV_PATH,
)


class DatasetManager:
    """Handles dataset validation, flexible column mapping, sanitization,

    and active dataset swapping for RAG & Analytics.
    """

    # Synonyms for auto-detecting column headers
    COLUMN_SYNONYMS = {
        "review_text": [
            "review_text", "review", "reviews", "comment", "comments",
            "feedback", "body", "text", "description", "content", "review_body"
        ],
        "rating": [
            "rating", "ratings", "stars", "star_rating", "score", "rate"
        ],
        "product_id": [
            "product_id", "product", "item_id", "asin", "sku",
            "product_name", "model", "phone", "device"
        ],
        "review_id": [
            "review_id", "id", "index", "uid"
        ]
    }

    # Backup of default dataset path
    DEFAULT_BACKUP_PATH = DATA_DIR / "reviews_default.csv"
    CUSTOM_CSV_PATH = DATA_DIR / "reviews_custom.csv"
    DATASET_METADATA_PATH = DATA_DIR / "active_dataset.json"

    def __init__(self):
        # Ensure default backup exists
        if REVIEWS_CSV_PATH.exists() and not self.DEFAULT_BACKUP_PATH.exists():
            shutil.copy(REVIEWS_CSV_PATH, self.DEFAULT_BACKUP_PATH)

    @classmethod
    def detect_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Map DataFrame columns to standard required columns using synonyms."""
        detected = {"review_text": None, "rating": None, "product_id": None, "review_id": None}
        lower_cols = {str(col).strip().lower(): col for col in df.columns}

        for standard_key, synonyms in cls.COLUMN_SYNONYMS.items():
            for syn in synonyms:
                if syn in lower_cols:
                    detected[standard_key] = lower_cols[syn]
                    break

        return detected

    @classmethod
    def clean_and_normalize(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate, normalize column names, scale ratings, and clean dirty rows."""
        total_incoming = len(df)
        detected = cls.detect_columns(df)

        if not detected["review_text"]:
            raise ValueError(
                "Could not find a review text column. "
                "Expected column names like: 'review_text', 'comment', 'feedback', or 'text'."
            )

        if not detected["rating"]:
            raise ValueError(
                "Could not find a rating column. "
                "Expected column names like: 'rating', 'stars', or 'score'."
            )

        # Build clean standard DataFrame
        clean_df = pd.DataFrame()

        # 1. Review Text
        clean_df["review_text"] = df[detected["review_text"]].astype(str).str.strip()

        # 2. Rating & Normalization
        raw_ratings = pd.to_numeric(df[detected["rating"]], errors="coerce")
        # If ratings are on 1-10 scale, normalize to 1-5
        max_rate = raw_ratings.max()
        if pd.notna(max_rate) and max_rate > 5:
            clean_df["rating"] = (raw_ratings / 2.0).round().astype("Int64")
        else:
            clean_df["rating"] = raw_ratings.round().astype("Int64")

        # 3. Product ID (Default to 'P101' if missing)
        if detected["product_id"]:
            clean_df["product_id"] = df[detected["product_id"]].astype(str).str.upper().str.strip()
        else:
            clean_df["product_id"] = "CATALOG-ITEM"

        # 4. Review ID
        if detected["review_id"]:
            clean_df["review_id"] = df[detected["review_id"]].astype(str).str.strip()
        else:
            clean_df["review_id"] = [str(i + 1) for i in range(len(clean_df))]

        # Filter out invalid / dirty rows
        initial_len = len(clean_df)
        clean_df = clean_df.dropna(subset=["rating", "review_text"])
        # Rating must be between 1 and 5
        clean_df = clean_df[(clean_df["rating"] >= 1) & (clean_df["rating"] <= 5)]
        # Filter out empty or trivial text (< 5 characters)
        clean_df = clean_df[clean_df["review_text"].str.len() >= 5]
        # Remove duplicate reviews
        clean_df = clean_df.drop_duplicates(subset=["review_text"])

        # Re-index review_ids sequentially if they became disjointed
        clean_df["review_id"] = [str(i + 1) for i in range(len(clean_df))]

        # Guardrail: Maximum 3,000 rows for fast local embedding generation
        dropped_count = initial_len - len(clean_df)
        if len(clean_df) > 3000:
            clean_df = clean_df.sample(n=3000, random_state=42).reset_index(drop=True)

        unique_products = sorted(clean_df["product_id"].unique().tolist())

        stats = {
            "incoming_rows": total_incoming,
            "valid_rows": len(clean_df),
            "dropped_rows": dropped_count,
            "unique_products": unique_products,
            "product_count": len(unique_products),
            "avg_rating": round(float(clean_df["rating"].mean()), 2) if len(clean_df) > 0 else 0.0,
            "detected_columns": detected
        }

        if len(clean_df) == 0:
            raise ValueError("All rows were filtered out due to invalid ratings or empty review text.")

        return clean_df, stats

    def activate_custom_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        retriever,
        analytics,
        chatbot
    ) -> Dict[str, Any]:
        """Clean, save as active reviews.csv, rebuild FAISS index, and reload analytics."""
        clean_df, stats = self.clean_and_normalize(df)

        # 1. Write clean CSV to REVIEWS_CSV_PATH
        clean_df.to_csv(REVIEWS_CSV_PATH, index=False)

        # 2. Rebuild FAISS index
        retriever.build_or_load_index(force_rebuild=True)

        # 3. Reload Analytics
        analytics.reload()

        # 4. Clear Chatbot History
        chatbot.clear_history()

        stats["active_dataset"] = filename
        stats["is_default"] = False
        return stats

    def restore_default_dataset(
        self,
        retriever,
        analytics,
        chatbot
    ) -> Dict[str, Any]:
        """Restore the original pre-packaged reviews dataset."""
        if self.DEFAULT_BACKUP_PATH.exists():
            shutil.copy(self.DEFAULT_BACKUP_PATH, REVIEWS_CSV_PATH)
        else:
            retriever.ensure_data_exists()

        # Rebuild FAISS index from restored default
        retriever.build_or_load_index(force_rebuild=True)
        analytics.reload()
        chatbot.clear_history()

        df = analytics.get_df()
        products = sorted(df["product_id"].unique().tolist())

        return {
            "active_dataset": "Default Catalog (P101-P110)",
            "is_default": True,
            "valid_rows": len(df),
            "product_count": len(products),
            "unique_products": products,
            "avg_rating": round(float(df["rating"].mean()), 2),
        }
