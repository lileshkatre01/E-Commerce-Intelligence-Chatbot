"""ReviewIQ Enterprise - Streamlit Dashboard

An interactive internal e-commerce intelligence UI for data science presentations.
Run via:
    streamlit run frontend/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from backend.chatbot import get_chatbot
from backend.analytics import get_analytics
from backend.retriever import get_retriever

# Configure Page
st.set_page_config(
    page_title="ReviewIQ Enterprise | E-Commerce Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Slate Enterprise Theme)
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; }
    .kpi-box {
        background-color: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin: 4px 0; }
    .kpi-sub { font-size: 0.72rem; color: #64748b; }
    .badge-alert { color: #ef4444; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Backend Instances
analytics = get_analytics()
retriever = get_retriever()
chatbot = get_chatbot()

# Sidebar: Controls & Filters
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=54)
st.sidebar.title("ReviewIQ Suite")
st.sidebar.caption("Enterprise Customer Review Intelligence")

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval Filters")

product_list = ["All Products"] + sorted(analytics.get_df()["product_id"].unique().tolist())
selected_product = st.sidebar.selectbox("Filter Product", product_list)
product_filter = None if selected_product == "All Products" else selected_product

rating_options = {
    "All Ratings": (None, None),
    "Complaints (1-2★ only)": (1, 2),
    "Positive (4-5★ only)": (4, 5),
    "5★ Only": (5, 5),
    "1★ Only": (1, 1),
}
selected_rating_label = st.sidebar.selectbox("Filter Rating", list(rating_options.keys()))
min_r, max_r = rating_options[selected_rating_label]

similarity_threshold = st.sidebar.slider(
    "Cosine Similarity Cutoff", min_value=0.10, max_value=0.80, value=0.35, step=0.05
)

top_k = st.sidebar.selectbox("Retrieved Reviews (Top-K)", [3, 5, 8, 10], index=1)

if st.sidebar.button("Clear Conversation"):
    chatbot.clear_history()
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Powered by FAISS • Gemini • Pandas")

# App Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("E-Commerce Intelligence Dashboard")
    st.caption("AI-powered customer review synthesis, sentiment analysis, and issue mining.")
with col_h2:
    st.success("Backend: Active | FAISS Indexed: 50")

# Top KPI Ribbon
summary = analytics.get_executive_summary()
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-title">Catalog Reviews</div>
            <div class="kpi-value">{summary['total_reviews']}</div>
            <div class="kpi-sub">Across {summary['total_products']} products (P101-P110)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-title">Average CSAT</div>
            <div class="kpi-value">{summary['overall_avg_rating']}★</div>
            <div class="kpi-sub">Overall customer rating</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    pos = summary['sentiment_overview']['positive_pct']
    neg = summary['sentiment_overview']['negative_pct']
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-title">Sentiment Split</div>
            <div class="kpi-value" style="color: #34d399;">{pos}% Pos</div>
            <div class="kpi-sub">{neg}% critical complaints</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    worst_p = summary['most_complained_product']
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-title">Issue Watch</div>
            <div class="kpi-value badge-alert">{worst_p['product_id']} ({worst_p['avg_rating']}★)</div>
            <div class="kpi-sub">{worst_p['negative_pct']}% negative feedback</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Tabs: Intelligence Chat & Analytics Deep Dive
tab_chat, tab_analytics = st.tabs(["💬 Intelligence Chatbot", "📊 Catalog Analytics & Topics"])

with tab_chat:
    st.markdown("##### Suggested Quick Inquiries:")
    quick_cols = st.columns(4)
    suggested_clicked = None
    if quick_cols[0].button("⚠️ Most Complained Product?"):
        suggested_clicked = "Which product is the most complained about and what are the issues?"
    if quick_cols[1].button("📈 Overall Sentiment Breakdown?"):
        suggested_clicked = "What is our overall customer sentiment and ratings breakdown?"
    if quick_cols[2].button("🔋 Best Battery Performance?"):
        suggested_clicked = "Which phone has the best battery life?"
    if quick_cols[3].button("🔍 P102 Defect Analysis"):
        suggested_clicked = "What are the common complaints for product P102?"

    # Session Messages State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Referenced Customer Reviews"):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")

    # Chat Input
    user_prompt = st.chat_input("Ask an analytical question regarding customer reviews...")
    prompt_to_run = suggested_clicked or user_prompt

    if prompt_to_run:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS embeddings and synthesizing analytics..."):
                try:
                    result = chatbot.ask(
                        question=prompt_to_run,
                        top_k=top_k,
                        product_id=product_filter,
                        min_rating=min_r,
                        max_rating=max_r,
                        similarity_threshold=similarity_threshold,
                    )
                    st.markdown(result["answer"])

                    if result.get("sources"):
                        with st.expander("📚 Grounded Sources Cited"):
                            for s in result["sources"]:
                                st.caption(f"• {s}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                        }
                    )
                except Exception as err:
                    st.error(f"Error querying chatbot: {err}")

with tab_analytics:
    st.subheader("Product CSAT Benchmark")
    products_data = analytics.get_product_stats()
    df_products = pd.DataFrame(products_data)

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        chart_df = df_products[["product_id", "avg_rating"]].sort_values("product_id")
        st.bar_chart(chart_df.set_index("product_id"))

    with col_table:
        st.dataframe(
            df_products[["product_id", "avg_rating", "positive_pct", "negative_pct", "status"]],
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Mined Complaint Categories")
    complaints = analytics.get_complaint_topics()
    complaints_list = []
    for topic, data in complaints["complaint_topics"].items():
        complaints_list.append(
            {
                "Issue Topic": topic,
                "Reviews Count": data["count"],
                "Catalog %": f"{data['pct_of_reviews']}%",
                "Affected Products": ", ".join(data["affected_products"]),
            }
        )
    st.table(pd.DataFrame(complaints_list))
