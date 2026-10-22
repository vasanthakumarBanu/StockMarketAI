/**
 * StockMarketAI: Frontend Dashboard Controller
 * Asynchronously interacts with FastAPI REST API endpoints
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global state
  let currentSymbol = "MSFT";
  let equityChart = null;

  // DOM Elements
  const symbolInput = document.getElementById("symbolInput");
  const predictBtn = document.getElementById("predictBtn");
  const chips = document.querySelectorAll(".chip");
  const tabs = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const statusPill = document.getElementById("systemStatusPill");
  const statusText = document.getElementById("systemStatusText");

  // Backtest Elements
  const btSymbolInput = document.getElementById("btSymbolInput");
  const btStrategySelect = document.getElementById("btStrategySelect");
  const btCapitalInput = document.getElementById("btCapitalInput");
  const btAtrSelect = document.getElementById("btAtrSelect");
  const runBacktestBtn = document.getElementById("runBacktestBtn");

  // Scanner Elements
  const scanWatchlistBtn = document.getElementById("scanWatchlistBtn");
  const scannerTableBody = document.getElementById("scannerTableBody");

  // Broker Auto-Execution Toggle Elements
  const navAutoTradeBtn = document.getElementById("navAutoTradeBtn");
  const navAutoTradeText = document.getElementById("navAutoTradeText");
  const autoExecutionToggleBtn = document.getElementById("autoExecutionToggleBtn");
  const autoExecutionToggleText = document.getElementById("autoExecutionToggleText");
  const brokerSelect = document.getElementById("brokerSelect");
  const executeBrokerOrderBtn = document.getElementById("executeBrokerOrderBtn");
  // Search & Company Elements
  const searchSuggestions = document.getElementById("searchSuggestions");
  const heroCompanyBanner = document.getElementById("heroCompanyBanner");
  const heroCompanyName = document.getElementById("heroCompanyName");
  const heroCompanyBadge = document.getElementById("heroCompanyBadge");

  // 1. Initialize Health Check & Auto-Execution Boolean State
  checkSystemHealth();
  fetchExecutionStatus();

  // 2. Fetch Initial Prediction for Default Symbol
  fetchPrediction(currentSymbol);

  // 3. Live Stock Autocomplete & Fuzzy Company Search (Indian & US Markets)
  let searchDebounceTimer = null;
  symbolInput.addEventListener("input", (e) => {
    const query = e.target.value.trim();
    clearTimeout(searchDebounceTimer);

    if (query.length < 2) {
      if (searchSuggestions) searchSuggestions.style.display = "none";
      return;
    }

    searchDebounceTimer = setTimeout(() => {
      fetchStockSuggestions(query);
    }, 250);
  });

  async function fetchStockSuggestions(query) {
    if (!searchSuggestions) return;
    try {
      const res = await fetch(`/api/v1/stocks/search?query=${encodeURIComponent(query)}&limit=7`);
      if (!res.ok) return;
      const items = await res.json();
      if (!items || items.length === 0) {
        searchSuggestions.style.display = "none";
        return;
      }
      renderSuggestions(items);
    } catch (_) {
      searchSuggestions.style.display = "none";
    }
  }

  function renderSuggestions(items) {
    if (!searchSuggestions) return;
    searchSuggestions.innerHTML = "";
    items.forEach(item => {
      const div = document.createElement("div");
      div.className = "suggestion-item";
      div.innerHTML = `
        <div class="suggestion-main">
          <span class="suggestion-symbol">${item.symbol}</span>
          <span class="suggestion-name">${item.name}</span>
        </div>
        <div class="suggestion-meta">
          <span class="suggestion-badge">${item.badge || item.market}</span>
        </div>
      `;
      div.addEventListener("click", () => {
        symbolInput.value = item.symbol;
        currentSymbol = item.symbol;
        btSymbolInput.value = item.symbol;
        searchSuggestions.style.display = "none";
        fetchPrediction(item.symbol);
      });
      searchSuggestions.appendChild(div);
    });
    searchSuggestions.style.display = "block";
  }

  // Hide suggestions dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (searchSuggestions && !searchSuggestions.contains(e.target) && e.target !== symbolInput) {
      searchSuggestions.style.display = "none";
    }
  });

  symbolInput.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && searchSuggestions) {
      searchSuggestions.style.display = "none";
    }
  });

  // 3. Setup Tab Navigation
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      // Auto-run scanner if clicking scanner tab for first time
      if (targetId === "tabScanner" && scannerTableBody.children.length === 0) {
        runWatchlistScan();
      }

      // Auto-run backtest if clicking backtesting tab for first time
      if (targetId === "tabBacktesting" && !equityChart) {
        runBacktest();
      }
    });
  });

  // 4. Quick Ticker Chips Click
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const sym = chip.getAttribute("data-symbol");
      symbolInput.value = sym;
      currentSymbol = sym;
      btSymbolInput.value = sym;
      fetchPrediction(sym);
    });
  });

  // 5. Prediction Search Button
  predictBtn.addEventListener("click", () => {
    if (searchSuggestions) searchSuggestions.style.display = "none";
    const sym = symbolInput.value.trim();
    if (sym) {
      currentSymbol = sym;
      btSymbolInput.value = sym;
      fetchPrediction(sym);
    }
  });

  symbolInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      predictBtn.click();
    }
  });

  // 6. Backtest Execute Button
  runBacktestBtn.addEventListener("click", () => {
    runBacktest();
  });

  // 7. Watchlist Scanner Button
  scanWatchlistBtn.addEventListener("click", () => {
    runWatchlistScan();
  });

  // =========================================================================
  // Health Check
  // =========================================================================
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/v1/health");
      if (res.ok) {
        const data = await res.json();
        statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
        statusPill.style.color = "var(--color-bullish)";
        statusText.textContent = `Online • ${data.models_loaded.feature_count} Features Ready`;
      } else {
        throw new Error("API Offline");
      }
    } catch (err) {
      statusPill.style.borderColor = "rgba(244, 63, 94, 0.4)";
      statusPill.style.color = "var(--color-bearish)";
      statusText.textContent = "Offline";
    }
  }

  // =========================================================================
  // Prediction Logic
  // =========================================================================
  async function fetchPrediction(symbol) {
    const loading = document.getElementById("predictionLoading");
    const results = document.getElementById("predictionResults");
    const loadingSymbol = document.getElementById("loadingSymbol");

    loadingSymbol.textContent = symbol;
    loading.style.display = "flex";
    results.style.opacity = "0.3";
    predictBtn.querySelector(".btn-loader").style.display = "inline-block";
    predictBtn.querySelector(".btn-text").textContent = "Predicting...";

    try {
      const res = await fetch(`/api/v1/predict/${encodeURIComponent(symbol)}?use_sentiment=true`);
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction request failed");
      }
      const data = await res.json();
      renderPrediction(data);
    } catch (err) {
      alert(`Error fetching prediction for ${symbol}: ${err.message}`);
    } finally {
      loading.style.display = "none";
      results.style.opacity = "1";
      predictBtn.querySelector(".btn-loader").style.display = "none";
      predictBtn.querySelector(".btn-text").textContent = "Generate AI Prediction";
    }
  }

  function renderPrediction(data) {
    const l1 = data.level1_direction;
    const l2 = data.level2_price_targets;
    const l3 = data.level3_trading_signal;
    const m = data.current_market;
    const s = data.sentiment;

    // --- Hero Banner: Level 3 Signal ---
    const heroCard = document.getElementById("signalHeroCard");
    const heroHeader = heroCard.querySelector(".signal-hero-header");
    const signalTitle = document.getElementById("signalTitle");
    const signalActionDesc = document.getElementById("signalActionDesc");
    const signalConfidenceText = document.getElementById("signalConfidenceText");

    signalTitle.textContent = l3.signal;
    signalActionDesc.textContent = l3.recommended_action;
    signalConfidenceText.textContent = `${Math.round(l3.confidence_pct)}%`;

    // Company Name & Market Badge
    if (heroCompanyBanner && heroCompanyName && heroCompanyBadge) {
      if (data.company_name) {
        heroCompanyBanner.style.display = "flex";
        heroCompanyName.textContent = data.company_name;
        heroCompanyBadge.textContent = `${data.badge || ""} ${data.symbol}`;
      } else {
        heroCompanyBanner.style.display = "none";
      }
    }

    // Dynamic color styling for signal
    heroHeader.className = "signal-hero-header";
    if (l3.signal.includes("BUY")) {
      heroHeader.classList.add("signal-buy");
    } else if (l3.signal.includes("SELL")) {
      heroHeader.classList.add("signal-sell");
    } else {
      heroHeader.classList.add("signal-hold");
    }

    // Hero metrics
    document.getElementById("metricCurrentPrice").textContent = `$${m.close.toFixed(2)}`;
    document.getElementById("metricPredictedClose").textContent = `$${l2.tomorrow_close.toFixed(2)}`;

    const changePill = document.getElementById("metricExpectedChange");
    const isPositive = l2.expected_change_pct >= 0;
    changePill.textContent = `${isPositive ? "+" : ""}${l2.expected_change_pct.toFixed(2)}%`;
    changePill.className = `change-pill ${isPositive ? "positive" : "negative"}`;

    document.getElementById("metricStopLoss").textContent = l3.stop_loss_price ? `$${l3.stop_loss_price.toFixed(2)}` : "N/A";
    document.getElementById("metricTakeProfit").textContent = l3.take_profit_price ? `$${l3.take_profit_price.toFixed(2)}` : "N/A";
    document.getElementById("metricRiskReward").textContent = `${l3.risk_reward_ratio}:1`;

    // --- Level 1: Direction ---
    const dirBadge = document.getElementById("directionBadge");
    const dirArrow = document.getElementById("directionArrow");
    const dirText = document.getElementById("directionText");
    const dirProb = document.getElementById("directionProb");
    const dirProbBar = document.getElementById("directionProbBar");

    const isUp = l1.prediction === "UP";
    dirText.textContent = l1.prediction;
    dirArrow.textContent = isUp ? "▲" : "▼";
    dirBadge.className = `direction-badge ${isUp ? "" : "down"}`;
    dirProb.textContent = `${l1.probability_pct.toFixed(1)}%`;
    dirProbBar.style.width = `${Math.min(Math.max(l1.probability_pct, 5), 95)}%`;

    document.getElementById("dirConfidenceVal").textContent = `${Math.round(l1.confidence * 100)}%`;
    document.getElementById("newsSentimentVal").textContent = `${s.sentiment_label} (${s.score >= 0 ? "+" : ""}${s.score.toFixed(2)})`;

    // --- Level 2: OHLC Targets ---
    document.getElementById("ohlcOpen").textContent = `$${l2.tomorrow_open.toFixed(2)}`;
    document.getElementById("ohlcHigh").textContent = `$${l2.tomorrow_high.toFixed(2)}`;
    document.getElementById("ohlcLow").textContent = `$${l2.tomorrow_low.toFixed(2)}`;
    document.getElementById("ohlcClose").textContent = `$${l2.tomorrow_close.toFixed(2)}`;

    document.getElementById("ohlcOpenToday").textContent = `Today: $${m.open.toFixed(2)}`;
    document.getElementById("ohlcHighToday").textContent = `Today: $${m.high.toFixed(2)}`;
    document.getElementById("ohlcLowToday").textContent = `Today: $${m.low.toFixed(2)}`;
    document.getElementById("ohlcCloseToday").textContent = `Today: $${m.close.toFixed(2)}`;

    document.getElementById("confidenceIntervalVal").textContent = `$${l2.interval_lower.toFixed(2)} — $${l2.interval_upper.toFixed(2)}`;
    document.getElementById("atrVal").textContent = `$${m.atr.toFixed(2)}`;

    // --- Technical Indicators Context ---
    document.getElementById("predictionDateText").textContent = `Date: ${data.prediction_date}`;
    document.getElementById("indRsi").textContent = m.rsi_14.toFixed(1);
    const rsiBadge = document.getElementById("indRsiBadge");
    if (m.rsi_14 >= 70) {
      rsiBadge.textContent = "Overbought";
      rsiBadge.style.color = "var(--color-bearish)";
    } else if (m.rsi_14 <= 30) {
      rsiBadge.textContent = "Oversold";
      rsiBadge.style.color = "var(--color-bullish)";
    } else {
      rsiBadge.textContent = "Neutral";
      rsiBadge.style.color = "var(--text-secondary)";
    }

    document.getElementById("indMacd").textContent = `${m.macd >= 0 ? "+" : ""}${m.macd.toFixed(2)}`;
    const macdBadge = document.getElementById("indMacdBadge");
    macdBadge.textContent = m.macd >= m.macd_signal ? "Bullish Cross" : "Bearish Cross";
    macdBadge.style.color = m.macd >= m.macd_signal ? "var(--color-bullish)" : "var(--color-bearish)";

    document.getElementById("indMa20").textContent = `$${m.ma_20.toFixed(2)}`;
    document.getElementById("indMa50").textContent = `$${m.ma_50.toFixed(2)}`;

    if (m.vwap !== undefined && m.vwap !== null) {
      document.getElementById("indVwap").textContent = `$${m.vwap.toFixed(2)}`;
      const vwapBadge = document.getElementById("indVwapBadge");
      const isAbove = m.close >= m.vwap;
      const vwapDist = m.vwap_distance !== undefined ? (m.vwap_distance * 100).toFixed(1) : ((m.close - m.vwap) / m.vwap * 100).toFixed(1);
      vwapBadge.textContent = isAbove ? `Above (${vwapDist >= 0 ? "+" : ""}${vwapDist}%)` : `Below (${vwapDist}%)`;
      vwapBadge.style.color = isAbove ? "var(--color-bullish)" : "var(--color-bearish)";
    }

    document.getElementById("indVol").textContent = `${(m.volatility_20 * 100).toFixed(1)}%`;
  }

  // =========================================================================
  // Backtesting Execution
  // =========================================================================
  async function runBacktest() {
    const sym = btSymbolInput.value.trim().toUpperCase() || "MSFT";
    const strategy = btStrategySelect.value;
    const capital = parseFloat(btCapitalInput.value) || 10000;
    const useAtr = btAtrSelect.value === "true";

    const loading = document.getElementById("btLoading");
    const resultsView = document.getElementById("btResultsView");
    loading.style.display = "flex";
    resultsView.style.opacity = "0.3";
    runBacktestBtn.querySelector(".btn-loader").style.display = "inline-block";
    runBacktestBtn.querySelector(".btn-text").textContent = "Simulating...";

    try {
      const res = await fetch("/api/v1/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: sym,
          strategy: strategy,
          initial_capital: capital,
          stop_loss: 0.03,
          take_profit: 0.06,
          use_atr: useAtr
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Backtest failed");
      }

      const data = await res.json();
      renderBacktest(data);
    } catch (err) {
      alert(`Backtest error: ${err.message}`);
    } finally {
      loading.style.display = "none";
      resultsView.style.opacity = "1";
      runBacktestBtn.querySelector(".btn-loader").style.display = "none";
      runBacktestBtn.querySelector(".btn-text").textContent = "Execute Backtest";
    }
  }

  function renderBacktest(data) {
    // Total Return
    const totalReturnElem = document.getElementById("btTotalReturn");
    const isPos = data.total_return_pct >= 0;
    totalReturnElem.textContent = `${isPos ? "+" : ""}${data.total_return_pct.toFixed(2)}%`;
    totalReturnElem.className = `mc-val ${isPos ? "positive" : "negative"}`;
    document.getElementById("btBenchmarkReturn").textContent = `Benchmark: +${data.benchmark_return_pct.toFixed(2)}%`;

    // Sharpe & Sortino
    document.getElementById("btSharpe").textContent = data.sharpe_ratio.toFixed(2);
    document.getElementById("btSortino").textContent = `Sortino: ${data.sortino_ratio.toFixed(2)}`;

    // Max Drawdown
    document.getElementById("btMaxDrawdown").textContent = `${data.max_drawdown_pct.toFixed(2)}%`;
    const diff = Math.abs(data.benchmark_drawdown_pct) / Math.max(Math.abs(data.max_drawdown_pct), 0.1);
    document.getElementById("btBenchDrawdown").textContent = `Benchmark: ${data.benchmark_drawdown_pct.toFixed(2)}% (~${diff.toFixed(1)}x safer!)`;

    // Win Rate & Trades
    document.getElementById("btWinRate").textContent = `${data.win_rate_pct.toFixed(1)}%`;
    document.getElementById("btTradesCount").textContent = `${data.winning_trades} wins / ${data.losing_trades} losses (${data.total_trades} total)`;

    // Profit Factor & Ending Equity
    document.getElementById("btProfitFactor").textContent = data.profit_factor.toFixed(2);
    document.getElementById("btFinalEquity").textContent = `$${data.final_equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    // Render Equity Curve Chart with Chart.js
    renderEquityCurveChart(data.equity_curve);

    // Render Executed Trades Table
    renderTradesTable(data.recent_trades);
  }

  function renderEquityCurveChart(curvePoints) {
    const ctx = document.getElementById("equityCurveChart").getContext("2d");
    const labels = curvePoints.map(p => p.date);
    const strategyData = curvePoints.map(p => p.strategy_equity);
    const benchmarkData = curvePoints.map(p => p.benchmark_equity);

    if (equityChart) {
      equityChart.destroy();
    }

    equityChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "StockMarketAI Strategy",
            data: strategyData,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.08)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.15,
            pointRadius: 2,
            pointHoverRadius: 6
          },
          {
            label: "Buy & Hold Benchmark",
            data: benchmarkData,
            borderColor: "rgba(148, 163, 184, 0.4)",
            borderWidth: 1.5,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
          },
          tooltip: {
            mode: "index",
            intersect: false,
            callbacks: {
              label: (context) => `${context.dataset.label}: $${context.raw.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(148, 163, 184, 0.06)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
          },
          y: {
            grid: { color: "rgba(148, 163, 184, 0.06)" },
            ticks: {
              color: "#64748b",
              font: { family: "JetBrains Mono", size: 11 },
              callback: (value) => `$${value.toLocaleString()}`
            }
          }
        }
      }
    });
  }

  function renderTradesTable(trades) {
    const tbody = document.getElementById("tradesTableBody");
    tbody.innerHTML = "";

    if (!trades || trades.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 24px;">No closed trades generated during this out-of-sample window.</td></tr>`;
      return;
    }

    trades.forEach(t => {
      const isWin = t.pnl >= 0;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${t.entry_date}</td>
        <td>${t.exit_date}</td>
        <td><span class="change-pill ${t.action === "BUY" ? "positive" : "negative"}">${t.action}</span></td>
        <td>$${t.entry_price.toFixed(2)}</td>
        <td>$${t.exit_price.toFixed(2)}</td>
        <td style="color: ${isWin ? "var(--color-bullish)" : "var(--color-bearish)"}; font-weight: 600;">
          ${isWin ? "+" : ""}$${t.pnl.toFixed(2)}
        </td>
        <td style="color: ${isWin ? "var(--color-bullish)" : "var(--color-bearish)"}; font-weight: 600;">
          ${isWin ? "+" : ""}${t.return_pct.toFixed(2)}%
        </td>
        <td><span class="ind-badge">${t.exit_reason}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // =========================================================================
  // Watchlist Scanner Logic
  // =========================================================================
  async function runWatchlistScan() {
    const loading = document.getElementById("scannerLoading");
    loading.style.display = "flex";
    scanWatchlistBtn.querySelector(".btn-loader").style.display = "inline-block";
    scanWatchlistBtn.querySelector(".btn-text").textContent = "Scanning...";

    try {
      const res = await fetch("/api/v1/stocks/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbols: ["MSFT", "AAPL", "NVDA", "TSLA", "AGYS", "GOOGL"]
        })
      });

      if (!res.ok) throw new Error("Watchlist scan failed");
      const scanResults = await res.json();
      renderScannerTable(scanResults);
    } catch (err) {
      alert(`Scanner error: ${err.message}`);
    } finally {
      loading.style.display = "none";
      scanWatchlistBtn.querySelector(".btn-loader").style.display = "none";
      scanWatchlistBtn.querySelector(".btn-text").textContent = "Scan Watchlist Now";
    }
  }

  function renderScannerTable(items) {
    scannerTableBody.innerHTML = "";
    items.forEach(item => {
      const isPos = item.expected_change_pct >= 0;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 700; color: #fff;">${item.symbol}</td>
        <td>$${item.current_price.toFixed(2)}</td>
        <td>$${item.predicted_close.toFixed(2)}</td>
        <td style="color: ${isPos ? "var(--color-bullish)" : "var(--color-bearish)"}; font-weight: 600;">
          ${isPos ? "+" : ""}${item.expected_change_pct.toFixed(2)}%
        </td>
        <td>
          <span style="color: ${item.direction === "UP" ? "var(--color-bullish)" : "var(--color-bearish)"}; font-weight: 700;">
            ${item.direction === "UP" ? "▲ UP" : "▼ DOWN"}
          </span>
        </td>
        <td>
          <span class="change-pill ${item.signal.includes("BUY") ? "positive" : item.signal.includes("SELL") ? "negative" : ""}">
            ${item.signal}
          </span>
        </td>
        <td>${Math.round(item.confidence)}%</td>
        <td style="color: var(--color-bearish);">${item.stop_loss ? `$${item.stop_loss.toFixed(2)}` : "-"}</td>
        <td style="color: var(--color-bullish);">${item.take_profit ? `$${item.take_profit.toFixed(2)}` : "-"}</td>
        <td>${item.rsi.toFixed(1)}</td>
        <td><span class="ind-badge">${item.sentiment}</span></td>
        <td>
          <button class="nav-link-btn" onclick="inspectSymbol('${item.symbol}')" style="padding: 4px 10px; font-size: 11px;">
            Inspect
          </button>
        </td>
      `;
      scannerTableBody.appendChild(tr);
    });
  }

  // =========================================================================
  // Master Broker Auto-Execution Boolean Switch Logic
  // =========================================================================
  if (autoExecutionToggleBtn) {
    autoExecutionToggleBtn.addEventListener("click", toggleAutoExecution);
  }
  if (navAutoTradeBtn) {
    navAutoTradeBtn.addEventListener("click", toggleAutoExecution);
  }
  if (executeBrokerOrderBtn) {
    executeBrokerOrderBtn.addEventListener("click", executeCurrentTradeOrder);
  }

  function updateExecutionUI(enabled) {
    autoExecutionEnabled = !!enabled;

    // Hero card toggle switch button
    if (autoExecutionToggleBtn && autoExecutionToggleText) {
      if (autoExecutionEnabled) {
        autoExecutionToggleBtn.className = "toggle-switch-btn on";
        autoExecutionToggleText.textContent = "AUTO-EXECUTION: ACTIVE (LIVE)";
      } else {
        autoExecutionToggleBtn.className = "toggle-switch-btn off";
        autoExecutionToggleText.textContent = "AUTO-EXECUTION: OFF (SAFE)";
      }
    }

    // Top navbar toggle switch button
    if (navAutoTradeBtn && navAutoTradeText) {
      if (autoExecutionEnabled) {
        navAutoTradeBtn.className = "nav-toggle-btn active";
        navAutoTradeText.textContent = "Auto-Trade: ON";
      } else {
        navAutoTradeBtn.className = "nav-toggle-btn safe";
        navAutoTradeText.textContent = "Auto-Trade: OFF";
      }
    }
  }

  async function fetchExecutionStatus() {
    try {
      const res = await fetch("/api/v1/brokers/execution-status");
      if (res.ok) {
        const data = await res.json();
        updateExecutionUI(data.auto_execution_enabled);
      }
    } catch (err) {
      console.warn("Could not fetch execution status", err);
    }
  }

  async function toggleAutoExecution() {
    const nextState = !autoExecutionEnabled;
    const selectedBroker = brokerSelect ? brokerSelect.value : "angel_one";

    try {
      const res = await fetch("/api/v1/brokers/toggle-execution", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: nextState,
          broker: selectedBroker,
          paper_trading: true
        })
      });

      if (res.ok) {
        const data = await res.json();
        updateExecutionUI(data.auto_execution_enabled);
        showExecutionNotice(
          data.auto_execution_enabled
            ? `⚡ Auto-Execution is now ENABLED (True). Live/Paper orders can now be executed via ${data.broker}.`
            : `🛡️ Auto-Execution is now DISABLED (False). All broker trades are safely blocked.`,
          data.auto_execution_enabled ? "success" : "info"
        );
      }
    } catch (err) {
      showExecutionNotice(`Failed to toggle auto-execution: ${err.message}`, "error");
    }
  }

  async function executeCurrentTradeOrder() {
    const sym = currentSymbol || "MSFT";
    const btnLoader = executeBrokerOrderBtn.querySelector(".btn-loader");
    const btnText = executeBrokerOrderBtn.querySelector(".btn-text");

    if (btnLoader) btnLoader.style.display = "inline-block";
    if (btnText) btnText.textContent = "Routing...";

    try {
      const res = await fetch("/api/v1/brokers/execute-now", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym })
      });

      const data = await res.json();

      if (data.status === "blocked") {
        showExecutionNotice(
          `⛔ Order Blocked: Master switch is OFF (auto_execution_enabled = false). Click the AUTO-EXECUTION button to turn it ON first.`,
          "warning"
        );
      } else if (data.status === "executed") {
        const o = data.order;
        showExecutionNotice(
          `✅ Order Placed #${o.order_id}: ${o.side} ${o.quantity} shares of ${o.symbol} @ $${o.price.toFixed(2)} (${o.status}).`,
          "success"
        );
      } else {
        showExecutionNotice(`ℹ️ ${data.message}`, "info");
      }
    } catch (err) {
      showExecutionNotice(`Order execution error: ${err.message}`, "error");
    } finally {
      if (btnLoader) btnLoader.style.display = "none";
      if (btnText) btnText.textContent = "Execute AI Order";
    }
  }

  function showExecutionNotice(message, type) {
    if (!executionNotice) return;
    executionNotice.style.display = "block";
    executionNotice.textContent = message;

    if (type === "warning" || type === "error") {
      executionNotice.style.borderColor = "rgba(244, 63, 94, 0.4)";
      executionNotice.style.color = "var(--color-bearish)";
    } else if (type === "success") {
      executionNotice.style.borderColor = "rgba(16, 185, 129, 0.4)";
      executionNotice.style.color = "var(--color-bullish)";
    } else {
      executionNotice.style.borderColor = "rgba(148, 163, 184, 0.3)";
      executionNotice.style.color = "var(--accent-cyan)";
    }
  }

  // Global helper to switch to prediction tab
  window.inspectSymbol = (sym) => {
    symbolInput.value = sym;
    currentSymbol = sym;
    btSymbolInput.value = sym;
    // Activate prediction tab
    const predTabBtn = document.querySelector('[data-tab="tabPrediction"]');
    if (predTabBtn) predTabBtn.click();
    fetchPrediction(sym);
  };
});
