# E-commerce Customer Review Intelligence Chatbot

A RAG (Retrieval-Augmented Generation) powered chatbot designed to analyze customer reviews, summarize customer sentiment, highlight recurring product issues, and answer product queries using Google's Gemini models and FAISS vector search.

---

## Project Structure

```
ecommerce-ai-chatbot/
│
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI API entry point & interactive CLI mode
│   ├── chatbot.py       # Manages prompt orchestration, chat history, and Gemini API calls
│   ├── retriever.py     # Embeddings, FAISS index loading/building, and review retrieval
│   └── config.py        # Centralized configurations, paths, and environment settings
│
├── data/
│   ├── reviews.csv      # Customer reviews dataset
│   └── faiss_index/     # Cached FAISS index and documents metadata
│       └── .gitkeep
│
├── frontend/            # Dedicated directory for Step 2 UI (Streamlit / React / Web UI)
│   └── README.md
│
├── .env                 # Environment variables (API keys)
├── .env.example         # Template for environment variables
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

---

## Module Responsibilities

- **`backend/config.py`**:
  Handles dynamic path resolution (`BASE_DIR`, `DATA_DIR`, `FAISS_INDEX_PATH`), loads environment variables via `python-dotenv`, and sets defaults for models and search parameters.

- **`backend/retriever.py`**:
  Encapsulates the `SentenceTransformer` (`all-MiniLM-L6-v2`) embedding model and `FAISS` vector database.
  - Loads or downloads `reviews.csv`.
  - Computes and caches vector embeddings on disk (`data/faiss_index/`) so expensive embedding generation only runs once.
  - Exposes `retrieve_context(query, k)` to retrieve the top `k` most relevant customer reviews.

- **`backend/chatbot.py`**:
  Encapsulates the `ReviewChatbot` class using the official `google-genai` SDK.
  - Maintains conversation history.
  - Injects retrieved review context into a structured system prompt.
  - Communicates with Gemini to provide grounded, conversational answers strictly based on customer feedback.

- **`backend/main.py`**:
  The application entry point:
  - **FastAPI API server**: Exposes REST endpoints (`/chat`, `/health`, `/reset`, `/history`, `/reindex`) with Swagger docs at `/docs`.
  - **CLI Mode**: Interactive terminal chat for quick testing without starting a frontend or browser.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Create and activate a virtual environment (recommended)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure your Gemini API Key
Create or edit your `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_TOP_K=5
```
You can obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

---

## How to Run

### Option A: Interactive Terminal CLI (Fastest for testing)
Test your chatbot directly from your terminal:
```powershell
python backend/main.py --cli
```
- Type your question (e.g., *"Which product has the best battery life?"* or *"What are common complaints about P102?"*).
- Type `reset` to clear chat history.
- Type `exit` or `quit` to exit.

### Option B: Enterprise Web Intelligence Dashboard (Recommended)
Start the unified FastAPI server:
```powershell
python backend/main.py
```
Or with `uvicorn`:
```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- Open **Enterprise Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
  - **Executive KPI Ribbon**: Total reviews (50), CSAT rating (4.32/5.0), sentiment health (86% positive), issue watch (P102).
  - **Interactive Metadata Filters**: Product selector (P101-P110), Star rating filter (Complaints 1-2★ only), Cosine threshold slider, Top-K selector.
  - **Clickable Suggested Prompts**: Fast analysis chips for sentiment breakdown, defect summaries, and battery evaluations.
  - **Interactive Review Citations**: Clicking any `[Review 7]` badge pops up a modal displaying the exact customer review text and metadata.
  - **Collapsible Analytics Panel**: Live Chart.js bar and doughnut charts visualizing product CSAT rankings and mined complaint topics.
- Open **API Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option C: Streamlit Companion App
For data science / notebook-style presentations:
```powershell
streamlit run frontend/streamlit_app.py
```

---

## API Endpoints

- **`POST /chat`**: Hybrid RAG + Analytics query endpoint.
- **`GET /analytics/summary`**: Global catalog KPIs (total reviews, overall average rating 4.32★, positive/neutral/negative split, highest-rated and most complained-about products).
- **`GET /analytics/products`**: Product-level analytics table (average rating, review volume, positive %, negative %, star distribution). Optional filter: `?product_id=P102`.
- **`GET /analytics/complaints`**: Automated NLP complaint topic mining across all reviews (topics, counts, percentages, and affected products).
- **`GET /health`**: System status and indexed document count.
- **`POST /reset`**: Reset conversation history.
- **`POST /reindex`**: Force rebuild of the vector index and reload analytics.


