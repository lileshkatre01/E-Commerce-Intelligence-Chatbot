import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.config import REVIEWS_CSV_PATH


class ReviewAnalytics:
    """E-commerce customer review analytics engine using Pandas and NLP rule-based topic mining."""

    COMPLAINT_TOPICS = {
        "Performance & Slowness": [
            r"\blag\b",
            r"\blags\b",
            r"\bslow\b",
            r"\bpoor performance\b",
            r"\bfrustrating\b",
        ],
        "Device Overheating": [
            r"\bheats up\b",
            r"\bhot\b",
            r"\boverheat\b",
            r"\bwarm\b",
        ],
        "Battery Drain & Short Battery": [
            r"\bdrain\b",
            r"\bdrains\b",
            r"\bbattery life is short\b",
            r"\bbattery drains\b",
        ],
        "Camera Limitations": [
            r"\bcamera quality is below\b",
            r"\bcamera is basic\b",
            r"\blow-light is average\b",
        ],
        "Physical Scratches & Build Wear": [
            r"\bscratch\b",
            r"\bscratched\b",
            r"\bchip\b",
            r"\bheavy\b",
        ],
        "Display Glare & Screen Issues": [
            r"\bscreen is average\b",
            r"\breflective\b",
            r"\bsunlight\b",
        ],
    }

    def __init__(self, csv_path: Path = REVIEWS_CSV_PATH):
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None

    def get_df(self) -> pd.DataFrame:
        """Load and cache the reviews DataFrame."""
        if self._df is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Reviews CSV not found at {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            df["product_id"] = df["product_id"].astype(str).str.upper().str.strip()
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
            df["review_text"] = df["review_text"].astype(str).fillna("")

            # Categorize sentiment based on standard e-commerce rating scale
            # Positive: 4-5 stars | Neutral: 3 stars | Negative: 1-2 stars
            def assign_sentiment(r: int) -> str:
                if r >= 4:
                    return "positive"
                elif r == 3:
                    return "neutral"
                else:
                    return "negative"

            df["sentiment"] = df["rating"].apply(assign_sentiment)
            self._df = df
        return self._df

    def reload(self) -> None:
        """Force reload from disk."""
        self._df = None
        self.get_df()

    def get_sentiment_summary(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute sentiment metrics and percentages globally or for a specific product."""
        df = self.get_df()
        if product_id:
            df = df[df["product_id"] == product_id.upper().strip()]

        total = len(df)
        if total == 0:
            return {
                "product_id": product_id,
                "total_reviews": 0,
                "avg_rating": 0.0,
                "positive_count": 0,
                "positive_pct": 0.0,
                "neutral_count": 0,
                "neutral_pct": 0.0,
                "negative_count": 0,
                "negative_pct": 0.0,
            }

        counts = df["sentiment"].value_counts().to_dict()
        pos = counts.get("positive", 0)
        neu = counts.get("neutral", 0)
        neg = counts.get("negative", 0)

        return {
            "product_id": product_id or "ALL",
            "total_reviews": total,
            "avg_rating": round(float(df["rating"].mean()), 2),
            "positive_count": pos,
            "positive_pct": round((pos / total) * 100, 1),
            "neutral_count": neu,
            "neutral_pct": round((neu / total) * 100, 1),
            "negative_count": neg,
            "negative_pct": round((neg / total) * 100, 1),
        }

    def get_product_stats(self) -> List[Dict[str, Any]]:
        """Return product performance ranking, ratings, and sentiment breakdown."""
        df = self.get_df()
        grouped = df.groupby("product_id")

        results = []
        for pid, group in grouped:
            total = len(group)
            avg_r = round(float(group["rating"].mean()), 2)
            counts = group["sentiment"].value_counts().to_dict()
            pos = counts.get("positive", 0)
            neu = counts.get("neutral", 0)
            neg = counts.get("negative", 0)

            # Rating distribution counts
            dist = {str(i): int((group["rating"] == i).sum()) for i in range(1, 6)}

            if neg > pos:
                status = "Mostly Critical"
            elif pos >= total * 0.8:
                status = "Highly Praised"
            else:
                status = "Mixed Reviews"

            results.append(
                {
                    "product_id": pid,
                    "total_reviews": total,
                    "avg_rating": avg_r,
                    "status": status,
                    "positive_pct": round((pos / total) * 100, 1),
                    "neutral_pct": round((neu / total) * 100, 1),
                    "negative_pct": round((neg / total) * 100, 1),
                    "rating_distribution": dist,
                }
            )

        # Sort descending by average rating, ascending by negative percentage
        results.sort(key=lambda x: (x["avg_rating"], -x["negative_pct"]), reverse=True)
        return results

    def get_complaint_topics(self) -> Dict[str, Any]:
        """Mine common complaint categories, frequencies, and affected products."""
        df = self.get_df()
        topic_data = {}

        for topic, patterns in self.COMPLAINT_TOPICS.items():
            regex = re.compile("|".join(patterns), re.IGNORECASE)
            matched_rows = df[df["review_text"].str.contains(regex, regex=True)]

            affected_products = sorted(list(matched_rows["product_id"].unique()))
            review_ids = [str(x) for x in matched_rows["review_id"].tolist()]

            topic_data[topic] = {
                "count": len(matched_rows),
                "pct_of_reviews": round((len(matched_rows) / len(df)) * 100, 1),
                "affected_products": affected_products,
                "review_ids": review_ids,
            }

        # Sort topics by frequency
        sorted_topics = dict(
            sorted(topic_data.items(), key=lambda item: item[1]["count"], reverse=True)
        )
        return {
            "total_reviews_analyzed": len(df),
            "complaint_topics": sorted_topics,
        }

    def get_executive_summary(self) -> Dict[str, Any]:
        """Provide a complete executive KPI dashboard."""
        df = self.get_df()
        sentiment = self.get_sentiment_summary()
        products = self.get_product_stats()
        complaints = self.get_complaint_topics()

        most_complained = min(products, key=lambda x: x["avg_rating"]) if products else None
        top_rated = max(products, key=lambda x: x["avg_rating"]) if products else None

        return {
            "total_reviews": len(df),
            "total_products": df["product_id"].nunique(),
            "overall_avg_rating": sentiment["avg_rating"],
            "sentiment_overview": {
                "positive_pct": sentiment["positive_pct"],
                "neutral_pct": sentiment["neutral_pct"],
                "negative_pct": sentiment["negative_pct"],
            },
            "top_performing_product": {
                "product_id": top_rated["product_id"] if top_rated else None,
                "avg_rating": top_rated["avg_rating"] if top_rated else None,
            },
            "most_complained_product": {
                "product_id": most_complained["product_id"] if most_complained else None,
                "avg_rating": most_complained["avg_rating"] if most_complained else None,
                "negative_pct": most_complained["negative_pct"] if most_complained else None,
            },
            "top_complaints": [
                {"topic": k, "count": v["count"], "products": v["affected_products"]}
                for k, v in list(complaints["complaint_topics"].items())[:3]
            ],
        }

    def generate_analytics_context(self, product_id: Optional[str] = None) -> str:
        """Format an informative analytics briefing text to include in Gemini's context."""
        summary = self.get_executive_summary()

        if product_id:
            p_stats = next(
                (p for p in self.get_product_stats() if p["product_id"] == product_id.upper()),
                None,
            )
            if p_stats:
                return (
                    f"Product-Specific Metrics for {p_stats['product_id']}:\n"
                    f"- Average Rating: {p_stats['avg_rating']}/5.0 stars\n"
                    f"- Sentiment Breakdown: {p_stats['positive_pct']}% Positive, "
                    f"{p_stats['neutral_pct']}% Neutral, {p_stats['negative_pct']}% Negative\n"
                    f"- Status: {p_stats['status']}\n"
                    f"- Star Breakdown: 5-star: {p_stats['rating_distribution']['5']}, "
                    f"4-star: {p_stats['rating_distribution']['4']}, "
                    f"3-star: {p_stats['rating_distribution']['3']}, "
                    f"2-star: {p_stats['rating_distribution']['2']}, "
                    f"1-star: {p_stats['rating_distribution']['1']}"
                )

        # Global metrics briefing
        top_complaint_lines = [
            f"  * {c['topic']}: {c['count']} reviews (Affected: {', '.join(c['products'])})"
            for c in summary["top_complaints"]
        ]
        return (
            f"Catalog E-Commerce Analytics Summary:\n"
            f"- Total Catalog Reviews: {summary['total_reviews']} across {summary['total_products']} products (P101-P110)\n"
            f"- Overall Catalog Rating: {summary['overall_avg_rating']} / 5.0 stars\n"
            f"- Global Sentiment: {summary['sentiment_overview']['positive_pct']}% Positive, "
            f"{summary['sentiment_overview']['neutral_pct']}% Neutral, "
            f"{summary['sentiment_overview']['negative_pct']}% Negative\n"
            f"- Most Complained-About Product: {summary['most_complained_product']['product_id']} "
            f"(Avg Rating: {summary['most_complained_product']['avg_rating']}/5.0, "
            f"{summary['most_complained_product']['negative_pct']}% Negative Reviews)\n"
            f"- Top Complaint Topics:\n" + "\n".join(top_complaint_lines)
        )


# Global singleton instance and functional helper
_default_analytics: Optional[ReviewAnalytics] = None


def get_analytics() -> ReviewAnalytics:
    """Get or initialize default ReviewAnalytics instance."""
    global _default_analytics
    if _default_analytics is None:
        _default_analytics = ReviewAnalytics()
    return _default_analytics
