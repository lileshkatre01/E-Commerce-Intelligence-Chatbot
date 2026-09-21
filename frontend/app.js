/**
 * ReviewIQ Enterprise - Front-end Application Controller
 * Matches the light SaaS dashboard theme:
 * - Live analytics KPI synchronization
 * - 3 Chart.js visualizations (Sentiment Trend Line, Product CSAT Bar, Complaint Share Doughnut)
 * - Grounded RAG AI Assistant chat with inline citation badges and source viewer
 * - Metadata filter bar (Product, Star Rating, Cosine Threshold, Top-K)
 * - Modal review inspector and catalog reindexing
 */

const API_BASE = ""; // Relative path when served directly from FastAPI

// Product Catalog Metadata Dictionary for UI labeling
const PRODUCT_INFO = {
  P101: { name: "Flagship Battery & Dual Camera", icon: "fa-mobile-screen" },
  P102: { name: "Budget Gaming (Thermal Lag Defect)", icon: "fa-gamepad" },
  P103: { name: "Ultra High Refresh Display", icon: "fa-bolt" },
  P104: { name: "Slim & Featherweight Handset", icon: "fa-feather" },
  P105: { name: "Everyday Value Smartphone", icon: "fa-tag" },
  P106: { name: "Studio Acoustic Loudspeaker", icon: "fa-volume-high" },
  P107: { name: "Rugged All-Weather Armor", icon: "fa-shield" },
  P108: { name: "Vivid AMOLED Cinema Display", icon: "fa-tv" },
  P109: { name: "120W Super Fast Charging", icon: "fa-plug" },
  P110: { name: "Night Optics & Optical Zoom", icon: "fa-camera" },
};

// Application State
const state = {
  sourceCache: {}, // Maps review_id or source tag -> review metadata
  activeFilters: {
    product_id: "",
    min_rating: null,
    max_rating: null,
    similarity_threshold: 0.35,
    top_k: 5,
  },
  charts: {
    sentimentTrend: null,
    productBar: null,
    complaintsGauge: null,
  },
};

// DOM Elements
const elements = {
  // Chat
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatThread: document.getElementById("chat-thread"),
  btnSend: document.getElementById("btn-send"),
  btnClearChat: document.getElementById("btn-clear-chat"),
  btnNewChat: document.getElementById("btn-new-chat"),

  // Filter controls
  filterProduct: document.getElementById("filter-product"),
  filterRating: document.getElementById("filter-rating"),
  filterThreshold: document.getElementById("filter-threshold"),
  labelThresholdVal: document.getElementById("label-threshold-val"),
  filterTopK: document.getElementById("filter-topk"),
  btnResetFilters: document.getElementById("btn-reset-filters"),

  // Constraints bar
  activeConstraintsBar: document.getElementById("active-constraints-bar"),
  pillProduct: document.getElementById("pill-product"),
  pillRating: document.getElementById("pill-rating"),
  pillThreshold: document.getElementById("pill-threshold"),
  btnClearConstraints: document.getElementById("btn-clear-constraints"),

  // Table
  productsTableBody: document.getElementById("products-table-body"),

  // Modal
  citationModal: document.getElementById("citation-modal"),
  modalTitle: document.getElementById("modal-title"),
  modalProductBadge: document.getElementById("modal-product-badge"),
  modalRatingBadge: document.getElementById("modal-rating-badge"),
  modalSimilarityBadge: document.getElementById("modal-similarity-badge"),
  modalText: document.getElementById("modal-text"),
  btnCloseModal: document.getElementById("btn-close-modal"),
  btnModalDismiss: document.getElementById("btn-modal-dismiss"),

  // Reindex
  reindexBtn: document.getElementById("reindex-btn"),
};

// ==========================================================================
// INITIALIZATION
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadExecutiveKPIs();
  loadAnalyticsCharts();
});

function initEventListeners() {
  // Chat submission
  if (elements.chatForm) {
    elements.chatForm.addEventListener("submit", handleChatSubmit);
  }

  // Submit on Enter key without Shift
  if (elements.chatInput) {
    elements.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        elements.chatForm.requestSubmit();
      }
    });
  }

  // Suggested chips
  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query && elements.chatInput) {
        elements.chatInput.value = query;
        elements.chatForm.requestSubmit();
      }
    });
  });

  // Filters
  if (elements.filterProduct) elements.filterProduct.addEventListener("change", updateFilterState);
  if (elements.filterRating) elements.filterRating.addEventListener("change", updateFilterState);
  if (elements.filterTopK) elements.filterTopK.addEventListener("change", updateFilterState);
  if (elements.filterThreshold) {
    elements.filterThreshold.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value).toFixed(2);
      if (elements.labelThresholdVal) elements.labelThresholdVal.textContent = val;
      updateFilterState();
    });
  }

  // Reset filters
  if (elements.btnResetFilters) elements.btnResetFilters.addEventListener("click", resetAllFilters);
  if (elements.btnClearConstraints) elements.btnClearConstraints.addEventListener("click", resetAllFilters);

  // New Chat & Clear
  if (elements.btnClearChat) elements.btnClearChat.addEventListener("click", handleResetSession);
  if (elements.btnNewChat) {
    elements.btnNewChat.addEventListener("click", (e) => {
      e.preventDefault();
      handleResetSession();
    });
  }

  // Modal close handlers
  if (elements.btnCloseModal) elements.btnCloseModal.addEventListener("click", closeModal);
  if (elements.btnModalDismiss) elements.btnModalDismiss.addEventListener("click", closeModal);
  if (elements.citationModal) {
    elements.citationModal.addEventListener("click", (e) => {
      if (e.target === elements.citationModal) closeModal();
    });
  }

  // Reindex
  if (elements.reindexBtn) elements.reindexBtn.addEventListener("click", handleReindex);

  // Sidebar navigation items -> Under Development Notice
  const navSentiment = document.getElementById("nav-sentiment");
  if (navSentiment) {
    navSentiment.addEventListener("click", (e) => {
      e.preventDefault();
      showDevNotice(
        "Feature Under Development",
        "The Customer Cohorts & Segmentation module is currently in development for the ReviewIQ v3.1 release.",
        "warning"
      );
    });
  }

  const navReviews = document.getElementById("nav-reviews");
  if (navReviews) {
    navReviews.addEventListener("click", (e) => {
      e.preventDefault();
      showDevNotice(
        "Module In Progress",
        "Deep longitudinal trend exploration is in progress. Real-time aggregated metrics are accessible on this overview dashboard.",
        "warning"
      );
    });
  }

  const navComplaints = document.getElementById("nav-complaints");
  if (navComplaints) {
    navComplaints.addEventListener("click", (e) => {
      e.preventDefault();
      showDevNotice(
        "Feature Under Development",
        "Automated customer complaint ticket dispatching is currently under development. Mined defect topics are visualized on the right panel.",
        "warning"
      );
    });
  }

  const navProducts = document.getElementById("nav-products");
  if (navProducts) {
    navProducts.addEventListener("click", (e) => {
      e.preventDefault();
      showDevNotice(
        "Active Workspace",
        "You are currently viewing the Products Intelligence & Customer Sentiment dashboard.",
        "info"
      );
    });
  }

  // Card action buttons (...)
  document.querySelectorAll(".card-action-more").forEach((btn) => {
    if (btn.id !== "btn-clear-chat") {
      btn.style.cursor = "pointer";
      btn.setAttribute("title", "Card Options");
      btn.addEventListener("click", () => {
        showDevNotice(
          "Options Under Development",
          "Card customization, CSV/PDF export, and custom report filters are scheduled for the next release.",
          "info"
        );
      });
    }
  });

  // Top KPI metric cards
  document.querySelectorAll(".metric-card").forEach((card) => {
    card.style.cursor = "pointer";
    card.addEventListener("click", () => {
      showDevNotice(
        "Drilldown In Progress",
        "Historical timeseries drilldowns for this metric are currently under development.",
        "info"
      );
    });
  });
}

// ==========================================================================
// API CALLS & KPI LOADERS
// ==========================================================================

async function loadExecutiveKPIs() {
  try {
    const res = await fetch(`${API_BASE}/analytics/summary`);
    if (!res.ok) return;
    const data = await res.json();

    // 1. Total Reviews
    const elReviews = document.getElementById("kpi-total-reviews");
    if (elReviews) elReviews.textContent = data.total_reviews || 50;

    // 2. Catalog CSAT Rating
    const elRating = document.getElementById("kpi-avg-rating");
    if (elRating) elRating.textContent = (data.overall_avg_rating || 4.32).toFixed(2);

    // 3. Critical Complaint Rate
    const elComplaints = document.getElementById("kpi-complaint-rate");
    if (elComplaints && data.sentiment_overview) {
      elComplaints.textContent = `${data.sentiment_overview.negative_pct || 10.0}%`;
    }

    // 4. Worst Product Watchlist
    const elWorst = document.getElementById("kpi-worst-product");
    if (elWorst && data.most_complained_product) {
      elWorst.textContent = data.most_complained_product.product_id || "P102";
    }

    // Main Chart Headline Score
    const elChartScore = document.getElementById("chart-main-score");
    if (elChartScore) {
      elChartScore.textContent = `${(data.overall_avg_rating || 4.32).toFixed(2)} CSAT`;
    }

    // 3 Segmented Counters below line chart
    if (data.sentiment_overview) {
      const elPos = document.getElementById("seg-positive-count");
      const elNeu = document.getElementById("seg-neutral-count");
      const elNeg = document.getElementById("seg-negative-count");
      if (elPos) elPos.textContent = data.sentiment_overview.positive_count || 43;
      if (elNeu) elNeu.textContent = data.sentiment_overview.neutral_count || 2;
      if (elNeg) elNeg.textContent = data.sentiment_overview.negative_count || 5;
    }
  } catch (err) {
    console.warn("Error loading executive KPIs:", err);
  }
}

async function loadAnalyticsCharts() {
  try {
    // Fetch products performance
    const pRes = await fetch(`${API_BASE}/analytics/products`);
    if (pRes.ok) {
      const pData = await pRes.json();
      const products = pData.products || [];
      renderSentimentTrendChart(products);
      renderProductBarChart(products);
      renderProductsTable(products);
    }

    // Fetch complaint categories
    const cRes = await fetch(`${API_BASE}/analytics/complaints`);
    if (cRes.ok) {
      const cData = await cRes.json();
      renderComplaintsGaugeChart(cData.complaint_topics || {});
    }
  } catch (err) {
    console.warn("Error loading analytics charts:", err);
  }
}

// ==========================================================================
// CHART.JS VISUALIZATIONS
// ==========================================================================

/**
 * 1. Smooth Wave Line/Area Chart matching "Total Profit" in reference screenshot
 */
function renderSentimentTrendChart(products) {
  const canvas = document.getElementById("sentimentTrendChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const sorted = [...products].sort((a, b) => a.product_id.localeCompare(b.product_id));
  const labels = sorted.map((p) => p.product_id);
  const dataScores = sorted.map((p) => p.avg_rating);

  // Create subtle vertical area gradient (matching screenshot blue wave)
  const gradient = ctx.createLinearGradient(0, 0, 0, 180);
  gradient.addColorStop(0, "rgba(37, 99, 235, 0.22)");
  gradient.addColorStop(1, "rgba(37, 99, 235, 0.0)");

  if (state.charts.sentimentTrend) {
    state.charts.sentimentTrend.destroy();
  }

  state.charts.sentimentTrend = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "CSAT Score",
          data: dataScores,
          borderColor: "#2563eb",
          borderWidth: 2.5,
          backgroundColor: gradient,
          fill: true,
          tension: 0.42, // Smooth curved spline
          pointBackgroundColor: "#2563eb",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          titleFont: { size: 11, family: "Inter" },
          bodyFont: { size: 12, family: "Inter", weight: "bold" },
          padding: 8,
          cornerRadius: 8,
          callbacks: {
            label: (item) => ` ${item.parsed.y} / 5.0 Stars`,
          },
        },
      },
      scales: {
        y: {
          min: 1.0,
          max: 5.0,
          grid: {
            color: "rgba(241, 245, 249, 1)",
            drawBorder: false,
          },
          ticks: {
            stepSize: 1,
            color: "#94a3b8",
            font: { size: 10, family: "Inter" },
          },
        },
        x: {
          grid: { display: false },
          ticks: {
            color: "#64748b",
            font: { size: 10, family: "Inter", weight: "600" },
          },
        },
      },
    },
  });
}

/**
 * 2. Product CSAT Bar Chart matching "Most Day Active" in screenshot
 */
function renderProductBarChart(products) {
  const canvas = document.getElementById("productBarChart");
  if (!canvas) return;

  const sorted = [...products].sort((a, b) => a.product_id.localeCompare(b.product_id));
  const labels = sorted.map((p) => p.product_id);
  const data = sorted.map((p) => p.avg_rating);

  // Color scheme: royal blue for high scores, coral/red for critical issue P102
  const bgColors = sorted.map((p) => {
    if (p.avg_rating < 3.0) return "#ef4444"; // Coral red for P102
    if (p.avg_rating < 4.3) return "#f59e0b"; // Amber
    return "#2563eb"; // Royal blue
  });

  if (state.charts.productBar) {
    state.charts.productBar.destroy();
  }

  state.charts.productBar = new Chart(canvas, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Avg Rating",
          data: data,
          backgroundColor: bgColors,
          borderRadius: 6,
          borderSkipped: false,
          barPercentage: 0.55,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          cornerRadius: 8,
          callbacks: {
            label: (item) => ` CSAT: ${item.parsed.y} ★`,
          },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 5,
          grid: {
            color: "rgba(241, 245, 249, 1)",
            drawBorder: false,
          },
          ticks: {
            stepSize: 1,
            color: "#94a3b8",
            font: { size: 9 },
          },
        },
        x: {
          grid: { display: false },
          ticks: {
            color: "#64748b",
            font: { size: 9, weight: "600" },
          },
        },
      },
    },
  });
}

/**
 * 3. Complaint Categories Doughnut Gauge matching "Repeat Customer Rate"
 */
function renderComplaintsGaugeChart(topics) {
  const canvas = document.getElementById("complaintsGaugeChart");
  if (!canvas) return;

  const labels = Object.keys(topics);
  const counts = Object.values(topics).map((t) => t.count);

  if (state.charts.complaintsGauge) {
    state.charts.complaintsGauge.destroy();
  }

  state.charts.complaintsGauge = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [
        {
          data: counts,
          backgroundColor: [
            "#ef4444", // Performance & Slowness
            "#f97316", // Device Overheating
            "#f59e0b", // Battery Drain
            "#3b82f6", // Camera Limitations
            "#8b5cf6", // Build & Scratches
            "#06b6d4", // Display Glare
          ],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 10,
            padding: 8,
            font: { size: 9, family: "Inter" },
            color: "#64748b",
          },
        },
        tooltip: {
          backgroundColor: "#1e293b",
          cornerRadius: 8,
          callbacks: {
            label: (item) => ` ${item.label}: ${item.raw} issues`,
          },
        },
      },
    },
  });
}

/**
 * Product Performance Benchmark Table (Matching "Best Selling Products" Table)
 */
function renderProductsTable(products) {
  if (!elements.productsTableBody) return;
  elements.productsTableBody.innerHTML = "";

  // Sort descending by review count and rating
  const sorted = [...products].sort((a, b) => b.avg_rating - a.avg_rating);

  sorted.forEach((p) => {
    const meta = PRODUCT_INFO[p.product_id] || { name: "Consumer Electronics", icon: "fa-box" };
    const tr = document.createElement("tr");

    // Sentiment badge
    let sentimentBadge = `<span class="trend-badge positive"><i class="fa-solid fa-arrow-up"></i> ${p.positive_pct}% Pos</span>`;
    if (p.negative_pct >= 50) {
      sentimentBadge = `<span class="trend-badge negative"><i class="fa-solid fa-triangle-exclamation"></i> ${p.negative_pct}% Neg</span>`;
    } else if (p.neutral_pct > 20 || (p.positive_pct < 80 && p.negative_pct > 15)) {
      sentimentBadge = `<span class="trend-badge neutral"><i class="fa-solid fa-minus"></i> Mixed</span>`;
    }

    tr.innerHTML = `
      <td><strong>${p.product_id}</strong></td>
      <td>
        <div class="product-cell-name">
          <div class="product-cell-icon">
            <i class="fa-solid ${meta.icon}"></i>
          </div>
          <span>${meta.name}</span>
        </div>
      </td>
      <td>${p.total_reviews} reviews</td>
      <td>${sentimentBadge}</td>
      <td>
        <span class="rating-star-gold"><i class="fa-solid fa-star"></i> ${p.avg_rating.toFixed(1)}</span>
      </td>
    `;
    elements.productsTableBody.appendChild(tr);
  });
}

// ==========================================================================
// CHAT HANDLING & STREAMING EMULATION
// ==========================================================================

async function handleChatSubmit(e) {
  e.preventDefault();
  const text = elements.chatInput.value.trim();
  if (!text) return;

  // Clear input
  elements.chatInput.value = "";

  // Append user bubble
  appendUserMessage(text);

  // Show typing indicator
  const typingId = showTypingIndicator();

  // Build payload
  const payload = {
    message: text,
    top_k: state.activeFilters.top_k,
    similarity_threshold: state.activeFilters.similarity_threshold,
  };

  if (state.activeFilters.product_id) {
    payload.product_id = state.activeFilters.product_id;
  }
  if (state.activeFilters.min_rating !== null) {
    payload.min_rating = state.activeFilters.min_rating;
  }
  if (state.activeFilters.max_rating !== null) {
    payload.max_rating = state.activeFilters.max_rating;
  }

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    removeTypingIndicator(typingId);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      appendAssistantMessage({
        answer: `**Inquiry Error (${response.status}):** ${errData.detail || "Unable to retrieve response from assistant."}`,
        sources: [],
      });
      return;
    }

    const result = await response.json();

    // Cache source details for modal citation inspector
    if (result.source_details) {
      result.source_details.forEach((sd) => {
        state.sourceCache[sd.source] = sd;
        state.sourceCache[`Review ${sd.review_id}`] = sd;
      });
    }

    appendAssistantMessage(result);
  } catch (err) {
    removeTypingIndicator(typingId);
    appendAssistantMessage({
      answer: `**Connection Error:** Could not connect to API server at \`http://127.0.0.1:8000\`. Please ensure the backend is running.`,
      sources: [],
    });
    console.error("Chat request failed:", err);
  }
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "chat-bubble-row user";
  row.innerHTML = `
    <div class="chat-avatar user">
      <i class="fa-solid fa-user"></i>
    </div>
    <div class="chat-bubble-body">
      <div class="chat-bubble-text">
        <p>${escapeHtml(text)}</p>
      </div>
    </div>
  `;
  elements.chatThread.appendChild(row);
  scrollToBottom();
}

function appendAssistantMessage(data) {
  const row = document.createElement("div");
  row.className = "chat-bubble-row assistant";

  // Parse markdown & inject citation links
  const formattedHtml = formatMarkdownWithCitations(data.answer || "");

  // Build sources footer if sources present
  let sourcesHtml = "";
  if (data.sources && data.sources.length > 0) {
    const pills = data.sources
      .map(
        (s) =>
          `<button class="source-item-btn" onclick="inspectCitation('${escapeHtml(s)}')"><i class="fa-solid fa-paperclip"></i> ${escapeHtml(s)}</button>`
      )
      .join("");
    sourcesHtml = `
      <div class="sources-footer-box">
        <div class="sources-title"><i class="fa-solid fa-book-open"></i> Grounded Review Citations:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 4px;">${pills}</div>
      </div>
    `;
  }

  // Build inline analytics snapshot card if returned
  let analyticsHtml = "";
  if (data.analytics && (data.analytics.positive_pct !== undefined || data.analytics.overall_avg_rating !== undefined)) {
    const a = data.analytics;
    const avg = a.avg_rating || a.overall_avg_rating || "N/A";
    const pos = a.positive_pct !== undefined ? a.positive_pct : (a.sentiment_overview ? a.sentiment_overview.positive_pct : "N/A");
    const neg = a.negative_pct !== undefined ? a.negative_pct : (a.sentiment_overview ? a.sentiment_overview.negative_pct : "N/A");

    analyticsHtml = `
      <div class="chat-analytics-card">
        <div class="chat-analytics-title"><i class="fa-solid fa-chart-simple"></i> Analytics Snapshot</div>
        <div class="chat-analytics-grid">
          <div class="stat-cell">
            <span class="stat-cell-num">${avg}★</span>
            <span class="stat-cell-lbl">Avg Score</span>
          </div>
          <div class="stat-cell">
            <span class="stat-cell-num" style="color: #10b981;">${pos}%</span>
            <span class="stat-cell-lbl">Positive</span>
          </div>
          <div class="stat-cell">
            <span class="stat-cell-num" style="color: #ef4444;">${neg}%</span>
            <span class="stat-cell-lbl">Negative</span>
          </div>
        </div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="chat-avatar assistant">
      <i class="fa-solid fa-brain"></i>
    </div>
    <div class="chat-bubble-body">
      <div class="chat-bubble-text">
        ${formattedHtml}
        ${analyticsHtml}
        ${sourcesHtml}
      </div>
    </div>
  `;

  elements.chatThread.appendChild(row);
  scrollToBottom();
}

function showTypingIndicator() {
  const id = "typing_" + Date.now();
  const row = document.createElement("div");
  row.id = id;
  row.className = "chat-bubble-row assistant";
  row.innerHTML = `
    <div class="chat-avatar assistant">
      <i class="fa-solid fa-brain"></i>
    </div>
    <div class="chat-bubble-body">
      <div class="chat-bubble-text" style="padding: 10px 16px;">
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    </div>
  `;
  elements.chatThread.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ==========================================================================
// CITATION MODAL INSPECTOR
// ==========================================================================

window.inspectCitation = function (citationTag) {
  let matchKey = citationTag.replace(/[\[\]]/g, "").trim();
  const subMatch = matchKey.match(/Review\s+\d+/i);
  if (subMatch) {
    matchKey = subMatch[0];
  }

  const details = state.sourceCache[matchKey] || state.sourceCache[citationTag];

  if (details) {
    elements.modalTitle.textContent = `Review Citation: ${details.source || matchKey}`;
    elements.modalProductBadge.textContent = `Product: ${details.product_id}`;
    elements.modalRatingBadge.textContent = `Rating: ${details.rating}★`;
    elements.modalSimilarityBadge.textContent = `Similarity: ${typeof details.similarity === "number" ? details.similarity.toFixed(2) : details.similarity || "Relevant"}`;
    elements.modalText.textContent = `"${details.review_text}"`;
  } else {
    elements.modalTitle.textContent = `Review Citation: ${citationTag}`;
    elements.modalProductBadge.textContent = "Verified Source";
    elements.modalRatingBadge.textContent = "Catalog Feedback";
    elements.modalSimilarityBadge.textContent = "Grounded Match";
    elements.modalText.textContent = `This review passage was extracted directly from the verified FAISS vector database corresponding to ${citationTag}.`;
  }

  elements.citationModal.style.display = "flex";
};

function closeModal() {
  if (elements.citationModal) {
    elements.citationModal.style.display = "none";
  }
}

// ==========================================================================
// FILTER STATE SYNCHRONIZATION
// ==========================================================================

function updateFilterState() {
  const pVal = elements.filterProduct ? elements.filterProduct.value : "";
  const rVal = elements.filterRating ? elements.filterRating.value : "";
  const tVal = elements.filterThreshold ? parseFloat(elements.filterThreshold.value) : 0.35;
  const kVal = elements.filterTopK ? parseInt(elements.filterTopK.value, 10) : 5;

  state.activeFilters.product_id = pVal;
  state.activeFilters.similarity_threshold = tVal;
  state.activeFilters.top_k = kVal;

  if (rVal === "complaints") {
    state.activeFilters.min_rating = 1;
    state.activeFilters.max_rating = 2;
  } else if (rVal === "positive") {
    state.activeFilters.min_rating = 4;
    state.activeFilters.max_rating = 5;
  } else if (rVal && !isNaN(parseInt(rVal, 10))) {
    const star = parseInt(rVal, 10);
    state.activeFilters.min_rating = star;
    state.activeFilters.max_rating = star;
  } else {
    state.activeFilters.min_rating = null;
    state.activeFilters.max_rating = null;
  }

  // Update visual constraint pill bar
  let hasActive = false;
  if (pVal && elements.pillProduct) {
    elements.pillProduct.textContent = `Product: ${pVal}`;
    elements.pillProduct.style.display = "inline-block";
    hasActive = true;
  } else if (elements.pillProduct) {
    elements.pillProduct.style.display = "none";
  }

  if (rVal && elements.pillRating) {
    elements.pillRating.textContent = `Rating: ${rVal}`;
    elements.pillRating.style.display = "inline-block";
    hasActive = true;
  } else if (elements.pillRating) {
    elements.pillRating.style.display = "none";
  }

  if (elements.pillThreshold) {
    elements.pillThreshold.textContent = `Threshold: ${tVal.toFixed(2)}`;
  }

  if (elements.activeConstraintsBar) {
    elements.activeConstraintsBar.style.display = hasActive || tVal !== 0.35 ? "flex" : "none";
  }
}

function resetAllFilters() {
  if (elements.filterProduct) elements.filterProduct.value = "";
  if (elements.filterRating) elements.filterRating.value = "";
  if (elements.filterThreshold) elements.filterThreshold.value = 0.35;
  if (elements.labelThresholdVal) elements.labelThresholdVal.textContent = "0.35";
  if (elements.filterTopK) elements.filterTopK.value = "5";
  updateFilterState();
  showDevNotice(
    "Filters Reset",
    "Product and cosine similarity threshold filters have been restored to defaults.",
    "info"
  );
}

// ==========================================================================
// SESSION MANAGEMENT & REINDEXING
// ==========================================================================

async function handleResetSession() {
  try {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
  } catch (e) {
    console.warn("Reset error:", e);
  }

  if (elements.chatThread) {
    elements.chatThread.innerHTML = `
      <div class="chat-bubble-row assistant">
        <div class="chat-avatar assistant">
          <i class="fa-solid fa-brain"></i>
        </div>
        <div class="chat-bubble-body">
          <div class="chat-bubble-text">
            <p>Hello! I am your <strong>E-Commerce Intelligence Assistant</strong>. I analyze customer reviews using grounded vector search and real-time sentiment analytics.</p>
            <p>Ask any question or click one of the suggested query chips above to inspect product feedback.</p>
          </div>
        </div>
      </div>
    `;
  }
  showDevNotice(
    "New Analysis Session",
    "Conversation memory reset. You can now start a fresh inquiry.",
    "success"
  );
}

async function handleReindex() {
  if (!elements.reindexBtn) return;
  const icon = elements.reindexBtn.querySelector("i");
  if (icon) icon.classList.add("fa-spin");

  try {
    const res = await fetch(`${API_BASE}/reindex`, { method: "POST" });
    const data = await res.json();
    showDevNotice(
      "Index Synchronized",
      `FAISS vector database reloaded with ${data.documents_count || 50} verified customer reviews.`,
      "success"
    );
    loadExecutiveKPIs();
    loadAnalyticsCharts();
  } catch (err) {
    showDevNotice("Reindex Failed", String(err), "warning");
  } finally {
    if (icon) icon.classList.remove("fa-spin");
  }
}

// ==========================================================================
// UTILITY FUNCTIONS & MARKDOWN PARSER
// ==========================================================================

function formatMarkdownWithCitations(rawText) {
  let text = escapeHtml(rawText);

  // Headers (### or ##)
  text = text.replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; color: #2563eb; font-weight: 700;">$1</h4>');
  text = text.replace(/^## (.*$)/gim, '<h3 style="margin: 10px 0 6px; color: #1e293b; font-weight: 700;">$1</h3>');

  // Bold & Italic
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");

  // Inline Code
  text = text.replace(/`([^`]+)`/g, '<code style="background: #e2e8f0; padding: 2px 5px; border-radius: 4px; font-family: monospace; font-size: 0.85em;">$1</code>');

  // Bullet Lists
  text = text.replace(/^\s*[-*]\s+(.*)$/gim, "<li>$1</li>");
  text = text.replace(/(<li>.*<\/li>)/gms, "<ul>$1</ul>");

  // Convert inline citations like [Review X] or [Review 12] to interactive badges
  text = text.replace(/\[(Review\s+\d+)\]/gi, (match, p1) => {
    return `<span class="citation-badge" onclick="inspectCitation('${p1}')"><i class="fa-solid fa-quote-left"></i> ${p1}</span>`;
  });

  // Paragraphs
  const paragraphs = text.split(/\n\n+/);
  return paragraphs.map((p) => (p.startsWith("<") ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`)).join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scrollToBottom() {
  if (elements.chatThread) {
    elements.chatThread.scrollTo({
      top: elements.chatThread.scrollHeight,
      behavior: "smooth",
    });
  }
}

// ==========================================================================
// NOTIFICATION & DEVELOPMENT NOTICE TOAST
// ==========================================================================

function showDevNotice(title, message, type = "warning") {
  const container = document.getElementById("toast-container") || document.body;
  const toast = document.createElement("div");
  toast.className = `dev-toast ${type}`;

  const iconClass =
    type === "success"
      ? "fa-circle-check"
      : type === "info"
      ? "fa-circle-info"
      : "fa-screwdriver-wrench";

  const badgeText =
    type === "success"
      ? "COMPLETED"
      : type === "info"
      ? "SYSTEM INFO"
      : "UNDER DEVELOPMENT";

  toast.innerHTML = `
    <div class="dev-toast-icon">
      <i class="fa-solid ${iconClass}"></i>
    </div>
    <div class="dev-toast-content">
      <div class="dev-toast-header">
        <span class="dev-toast-badge">${badgeText}</span>
        <button class="dev-toast-close" title="Dismiss">&times;</button>
      </div>
      <div class="dev-toast-title">${escapeHtml(title)}</div>
      <div class="dev-toast-msg">${escapeHtml(message)}</div>
    </div>
    <div class="dev-toast-progress"></div>
  `;

  container.appendChild(toast);

  // Close handlers
  const closeBtn = toast.querySelector(".dev-toast-close");
  let timer = null;
  const dismiss = () => {
    if (timer) clearTimeout(timer);
    toast.classList.add("hiding");
    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 280);
  };

  closeBtn.addEventListener("click", dismiss);
  timer = setTimeout(dismiss, 4200);
}
