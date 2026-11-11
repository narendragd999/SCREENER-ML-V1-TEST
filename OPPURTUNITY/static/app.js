/**
 * app.js — Frontend Application Logic for Opportunity Engine
 */

let allOpportunities = [];
let currentFilterType = "MACRO_FLOOR";

function filterBySetupType(type) {
  currentFilterType = type;
  updatePillStyles();

  if (!allOpportunities || allOpportunities.length === 0) return;

  let filtered = allOpportunities;
  if (type === "MACRO_FLOOR") {
    filtered = allOpportunities.filter(o => o.is_at_macro_floor || (o.gain_probability_rating && (o.gain_probability_rating.startsWith("VERY HIGH") || (o.gain_probability_rating.startsWith("HIGH") && o.range_percentile <= 35))));
  } else if (type === "AT_STRUCTURAL_LOW") {
    filtered = allOpportunities.filter(o => o.is_at_structural_low || o.is_at_macro_floor);
  } else if (type === "STRUCTURAL_REBOUND") {
    filtered = allOpportunities.filter(o => o.structural_status === "REBOUNDING_FROM_STRUCTURAL_LOW" || o.is_rebounding_from_macro);
  } else if (type === "AT_SUPPORT_FLOOR") {
    filtered = allOpportunities.filter(o => o.is_on_support);
  } else if (type === "PRIME") {
    filtered = allOpportunities.filter(o => o.is_prime);
  } else if (type === "PENDING") {
    filtered = allOpportunities.filter(o => o.status !== "TARGET_HIT");
  }

  renderOpportunities(filtered);
}

function updatePillStyles() {
  const pills = {
    "MACRO_FLOOR": document.getElementById("pill-macro-floor"),
    "AT_STRUCTURAL_LOW": document.getElementById("pill-structural-at"),
    "ALL": document.getElementById("pill-all"),
    "STRUCTURAL_REBOUND": document.getElementById("pill-structural-reb"),
    "AT_SUPPORT_FLOOR": document.getElementById("pill-support"),
    "PRIME": document.getElementById("pill-prime"),
    "PENDING": document.getElementById("pill-pending"),
  };

  for (const [key, el] of Object.entries(pills)) {
    if (!el) continue;
    if (key === currentFilterType) {
      if (key === "MACRO_FLOOR") {
        el.className = "filter-pill px-3.5 py-1.5 rounded-xl text-xs font-black bg-emerald-500 text-black shadow-lg shadow-emerald-500/20 transition-all flex items-center gap-1.5 ring-2 ring-emerald-300";
      } else if (key === "AT_STRUCTURAL_LOW") {
        el.className = "filter-pill px-3.5 py-1.5 rounded-xl text-xs font-bold bg-amber-500 text-black shadow-lg shadow-amber-500/20 transition-all flex items-center gap-1.5 ring-2 ring-amber-300";
      } else {
        el.className = "filter-pill px-3 py-1.5 rounded-xl text-xs font-semibold bg-brand-600 text-white shadow-md transition-all flex items-center gap-1.5";
      }
    } else {
      if (key === "MACRO_FLOOR") {
        el.className = "filter-pill px-3.5 py-1.5 rounded-xl text-xs font-extrabold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 hover:bg-emerald-500/30 transition-all flex items-center gap-1.5";
      } else if (key === "AT_STRUCTURAL_LOW") {
        el.className = "filter-pill px-3.5 py-1.5 rounded-xl text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 transition-all flex items-center gap-1.5";
      } else {
        el.className = "filter-pill px-3 py-1.5 rounded-xl text-xs font-semibold bg-dark-750 text-gray-300 border border-gray-700 hover:bg-dark-700 transition-all flex items-center gap-1.5";
      }
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    
    // Update market status badge
    const regimeEl = document.getElementById("market-regime-text");
    if (regimeEl) {
      regimeEl.textContent = `${data.market_regime} (${data.market_as_of || 'Today'})`;
      if (data.market_regime === "UPTREND") {
        regimeEl.className = "font-semibold text-brand-400";
      } else if (data.market_regime === "DOWNTREND") {
        regimeEl.className = "font-semibold text-red-400";
      } else {
        regimeEl.className = "font-semibold text-yellow-400";
      }
    }

    const universeEl = document.getElementById("universe-count-text");
    if (universeEl) {
      universeEl.textContent = `${data.vetted_tickers_count} Healthy Stocks`;
    }

    // Refresh tracker count badge
    loadTrackerPositions(false);
  } catch (err) {
    console.error("Initialization error:", err);
  }
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.remove("border-brand-500", "text-brand-400");
    btn.classList.add("border-transparent", "text-gray-400");
  });

  const activeContent = document.getElementById(tabId);
  const activeBtn = document.getElementById("btn-" + tabId);

  if (activeContent) activeContent.classList.remove("hidden");
  if (activeBtn) {
    activeBtn.classList.remove("border-transparent", "text-gray-400");
    activeBtn.classList.add("border-brand-500", "text-brand-400");
  }

  if (tabId === "tab-tracker") {
    loadTrackerPositions(true);
  }
}

// ---------------- SCANNER LOGIC ----------------
async function triggerScan() {
  const scope = document.getElementById("scan-scope").value;
  const targetPct = document.getElementById("scan-target-pct").value;
  const supportOnly = document.getElementById("scan-support-only").checked;
  const uptrendOnly = document.getElementById("scan-uptrend-only").checked;
  const pendingOnly = document.getElementById("scan-pending-only") ? document.getElementById("scan-pending-only").checked : false;

  const btn = document.getElementById("btn-scan");
  const loading = document.getElementById("scan-loading");
  const container = document.getElementById("opportunities-container");

  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Scanning...`;
  loading.classList.remove("hidden");
  container.innerHTML = "";

  try {
    const url = `/api/scan?limit=${scope}&target_pct=${targetPct}&support_only=${supportOnly}&uptrend_only=${uptrendOnly}&pending_only=${pendingOnly}`;
    const res = await fetch(url);
    const data = await res.json();

    loading.classList.add("hidden");
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Scan Opportunities`;

    // Render Stats
    document.getElementById("stat-total-found").textContent = data.opportunities_count || 0;
    if (document.getElementById("stat-macro-floor-count")) {
      document.getElementById("stat-macro-floor-count").textContent = data.macro_at_count || 0;
    }
    document.getElementById("stat-structural-at-count").textContent = data.structural_at_count || 0;
    document.getElementById("stat-support-count").textContent = data.support_count || 0;
    document.getElementById("stat-prime-count").textContent = data.prime_count || 0;

    allOpportunities = data.opportunities || [];

    // Update Quick Filter Pill Counts
    const macroFloorCount = allOpportunities.filter(o => o.is_at_macro_floor || (o.gain_probability_rating && (o.gain_probability_rating.startsWith("VERY HIGH") || (o.gain_probability_rating.startsWith("HIGH") && o.range_percentile <= 35)))).length;
    const structAtCount = allOpportunities.filter(o => o.is_at_structural_low || o.is_at_macro_floor).length;
    const structRebCount = allOpportunities.filter(o => o.structural_status === "REBOUNDING_FROM_STRUCTURAL_LOW" || o.is_rebounding_from_macro).length;
    const supportCount = allOpportunities.filter(o => o.is_on_support).length;
    const primeCount = allOpportunities.filter(o => o.is_prime).length;
    const pendingCount = allOpportunities.filter(o => o.status !== "TARGET_HIT").length;

    if (document.getElementById("count-pill-macro-floor")) {
      document.getElementById("count-pill-macro-floor").textContent = macroFloorCount;
    }
    document.getElementById("count-pill-structural-at").textContent = structAtCount;
    document.getElementById("count-pill-all").textContent = allOpportunities.length;
    document.getElementById("count-pill-structural-reb").textContent = structRebCount;
    document.getElementById("count-pill-support").textContent = supportCount;
    document.getElementById("count-pill-prime").textContent = primeCount;
    document.getElementById("count-pill-pending").textContent = pendingCount;

    // Default to MACRO_FLOOR if stocks are at rock-bottom floor, else AT_STRUCTURAL_LOW, else ALL
    currentFilterType = macroFloorCount > 0 ? "MACRO_FLOOR" : (structAtCount > 0 ? "AT_STRUCTURAL_LOW" : "ALL");
    updatePillStyles();

    filterBySetupType(currentFilterType);
  } catch (err) {
    loading.classList.add("hidden");
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Scan Opportunities`;
    container.innerHTML = `
      <div class="p-6 bg-red-950/30 border border-red-800 rounded-2xl text-center">
        <p class="text-sm font-semibold text-red-400">Scan failed: ${err.message}</p>
      </div>
    `;
  }
}

function renderOpportunities(opps) {
  const container = document.getElementById("opportunities-container");
  if (!opps || opps.length === 0) {
    container.innerHTML = `
      <div class="text-center py-12 bg-dark-800/40 border border-gray-800 rounded-2xl">
        <i class="fa-solid fa-filter-circle-xmark text-4xl text-gray-600 mb-3"></i>
        <h4 class="text-sm font-semibold text-gray-300">No qualifying opportunities found for selected filter</h4>
        <p class="text-xs text-gray-500 mt-1">Try clicking "All Opportunities" or expanding your scan scope.</p>
      </div>
    `;
    return;
  }

  let html = "";
  opps.forEach(opp => {
    const sym = opp.symbol;
    const sec = opp.security || sym;
    const price = opp.current_price;
    const ema = opp.ema9;
    const target = opp.target_price;
    const targetPct = opp.target_pct;
    const distTarget = opp.distance_to_target_pct;
    const score = opp.score;
    const isPrime = opp.is_prime;
    const isOnSupport = opp.is_on_support;
    const isStructural = opp.has_structural_low;
    const isAtStructural = opp.is_at_structural_low;
    const isAtMacro = opp.is_at_macro_floor;

    // 0. Powerful Macro Rock-Bottom Floor Badge (Highest Gain Probability)
    let macroBadge = "";
    if (isAtMacro) {
      const proxText = opp.macro_proximity_pct > 0 ? `+${opp.macro_proximity_pct}%` : `${opp.macro_proximity_pct}%`;
      macroBadge = `
        <span class="px-3 py-1 rounded-xl text-xs font-black bg-emerald-500/20 text-emerald-300 border-2 border-emerald-500/70 flex items-center gap-1.5 shadow-lg shadow-emerald-500/20 ring-1 ring-emerald-400">
          <i class="fa-solid fa-bolt text-emerald-400"></i> ⚡ ROCK-BOTTOM FLOOR ₹${opp.macro_floor_price} (${proxText} • Prev Surge: +${opp.macro_rally_pct}% • Upside: +${opp.upside_to_cycle_high}%)
        </span>
      `;
    } else if (opp.gain_probability_rating && (opp.gain_probability_rating.startsWith("VERY HIGH") || opp.gain_probability_rating.startsWith("HIGH"))) {
      macroBadge = `
        <span class="px-2.5 py-1 rounded-lg text-xs font-bold bg-emerald-950/60 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5">
          <i class="fa-solid fa-bolt text-emerald-400"></i> High Gain Prob (Floor: ₹${opp.macro_floor_price || 'N/A'}, Upside: +${opp.upside_to_cycle_high}%)
        </span>
      `;
    }

    // 1. Structural Low Support Pill (PRIME FOCUS)
    let supportBadge = "";
    if (isAtStructural) {
      const proxText = opp.structural_proximity_pct > 0 ? `+${opp.structural_proximity_pct}%` : `${opp.structural_proximity_pct}%`;
      supportBadge = `
        <span class="px-3 py-1 rounded-xl text-xs font-extrabold bg-amber-500/20 text-amber-300 border border-amber-500/60 flex items-center gap-1.5 shadow-lg shadow-amber-500/10 ring-1 ring-amber-400">
          <i class="fa-solid fa-anchor"></i> ★ STRUCTURAL LOW ₹${opp.structural_low_price} (${proxText} • Prev Surge: +${opp.structural_rally_pct}%)
        </span>
      `;
    } else if (opp.structural_status === "REBOUNDING_FROM_STRUCTURAL_LOW") {
      supportBadge = `
        <span class="px-2.5 py-1 rounded-lg text-xs font-bold bg-cyan-950/60 text-cyan-300 border border-cyan-500/40 flex items-center gap-1.5">
          <i class="fa-solid fa-arrow-trend-up"></i> Rebounding off Structural Low ₹${opp.structural_low_price} (+${opp.structural_proximity_pct}%)
        </span>
      `;
    } else if (opp.support_status === "AT_SUPPORT_FLOOR") {
      const proxText = opp.support_proximity_pct > 0 ? `+${opp.support_proximity_pct}%` : `${opp.support_proximity_pct}%`;
      supportBadge = `
        <span class="px-2.5 py-1 rounded-lg text-xs font-bold bg-blue-600 text-white flex items-center gap-1.5 shadow-md shadow-blue-500/30 ring-1 ring-blue-400">
          <i class="fa-solid fa-layer-group"></i> Support Floor ₹${opp.support_level} (${opp.support_touches} touches, ${proxText})
        </span>
      `;
    } else if (isOnSupport) {
      supportBadge = `
        <span class="px-2.5 py-1 rounded-lg text-xs font-semibold badge-support text-white flex items-center gap-1.5 shadow-md shadow-blue-500/20">
          <i class="fa-solid fa-arrows-to-dot"></i> Support Floor ₹${opp.support_level} (${opp.support_touches} touches)
        </span>
      `;
    }

    // Fair value pill
    let fvBadge = "";
    if (opp.fair_gap_pct !== null && opp.fair_gap_pct !== undefined) {
      const isPositive = opp.fair_gap_pct >= 0;
      const color = isPositive ? "text-brand-400 bg-brand-500/20 border-brand-500/30" : "text-yellow-400 bg-yellow-500/20 border-yellow-500/30";
      fvBadge = `
        <span class="px-2 py-0.5 rounded-md text-xs font-medium border ${color}">
          Fair Val: ₹${opp.composite_fair_price || 'N/A'} (${isPositive ? '+' : ''}${opp.fair_gap_pct}%)
        </span>
      `;
    }

    // Status pill
    let statusClass = "bg-gray-700 text-gray-300";
    let statusText = opp.status;
    if (isAtMacro) {
      statusClass = "bg-emerald-500 text-black font-black shadow-md shadow-emerald-500/30";
      statusText = "⚡ AT ROCK-BOTTOM FLOOR";
    } else if (isAtStructural) {
      statusClass = "bg-amber-500/20 text-amber-300 border border-amber-500/50 font-extrabold";
      statusText = "★ AT STRUCTURAL LOW";
    } else if (opp.status === "REBOUNDING" && isStructural) {
      statusClass = "bg-cyan-600/30 text-cyan-300 border border-cyan-400 font-bold";
      statusText = "REBOUNDING OFF STRUCTURAL LOW";
    } else if (opp.status === "AT_SUPPORT") {
      statusClass = "bg-blue-600/30 text-blue-300 border border-blue-400 font-bold";
      statusText = "AT SUPPORT FLOOR";
    } else if (opp.status === "REBOUNDING") {
      statusClass = "bg-cyan-600/30 text-cyan-300 border border-cyan-400 font-semibold";
      statusText = "REBOUNDING OFF FLOOR";
    } else if (opp.status === "FRESH") {
      statusClass = "bg-brand-500/20 text-brand-400 border border-brand-500/30 font-semibold";
    } else if (opp.status === "IN_PROFIT") {
      statusClass = "bg-blue-500/20 text-blue-400 border border-blue-500/30";
    } else if (opp.status === "IN_DRAWDOWN") {
      statusClass = "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30";
    } else if (opp.status === "TARGET_HIT") {
      statusClass = "bg-green-600 text-white font-bold";
      statusText = "TARGET HIT ✓";
    }

    let setupSubline = "";
    if (isAtMacro) {
      setupSubline = `Current: <span class="font-bold text-white">₹${price}</span> • ⚡ Rock-Bottom Cycle Floor: <span class="font-black text-emerald-400">₹${opp.macro_floor_price}</span> (${opp.macro_proximity_pct > 0 ? '+' : ''}${opp.macro_proximity_pct}%) • <span class="text-emerald-300 font-semibold">Prev Bull Expansion: +${opp.macro_rally_pct}%</span> • <span class="text-brand-400 font-bold">Upside to Peak: +${opp.upside_to_cycle_high}%</span> • Defended: ${(opp.macro_dates || []).slice(-2).join(", ")}`;
    } else if (isAtStructural) {
      setupSubline = `Current: <span class="font-bold text-white">₹${price}</span> • Major Structural Low: <span class="font-extrabold text-amber-400">₹${opp.structural_low_price}</span> (${opp.structural_proximity_pct > 0 ? '+' : ''}${opp.structural_proximity_pct}% away) • Rock-Bottom Floor: <span class="font-bold text-emerald-400">₹${opp.macro_floor_price}</span> (+${opp.upside_to_cycle_high}% to peak) • Defended: ${(opp.structural_dates || []).slice(-2).join(", ")}`;
    } else if (opp.structural_status === "REBOUNDING_FROM_STRUCTURAL_LOW") {
      setupSubline = `Current: <span class="font-bold text-white">₹${price}</span> • Structural Floor: <span class="font-semibold text-cyan-400">₹${opp.structural_low_price}</span> • Rock-Bottom: ₹${opp.macro_floor_price} • <span class="text-brand-400 font-semibold">Historical Expansion: +${opp.structural_rally_pct}%</span> • Confirmed Rebound`;
    } else if (opp.setup_type === "AT_SUPPORT_FLOOR") {
      setupSubline = `Current: <span class="font-semibold text-white">₹${price}</span> • Support Floor: <span class="font-semibold text-blue-400">₹${opp.support_level}</span> (${opp.support_touches} multi-year touches) • <span class="text-blue-300 font-medium">Accumulation at Floor</span>`;
    } else if (opp.setup_type === "EMA_BREAKOUT_AT_SUPPORT") {
      setupSubline = `Current: <span class="font-semibold text-white">₹${price}</span> • 9-EMA: ₹${ema} • Support Floor: <span class="font-semibold text-blue-400">₹${opp.support_level}</span> • Confirmed: ${opp.confirm_date}`;
    } else {
      setupSubline = `Current: <span class="font-semibold text-white">₹${price}</span> • 9-EMA: ₹${ema} • Confirmed: ${opp.confirm_date} (${opp.candles_ago}d ago)`;
    }

    const cardBorder = isAtMacro 
      ? 'border-2 border-emerald-500/70 shadow-xl shadow-emerald-500/10 ring-1 ring-emerald-500/30' 
      : (isAtStructural 
          ? 'border-amber-500/50 shadow-lg shadow-amber-500/5 ring-1 ring-amber-500/20' 
          : (isPrime ? 'border-brand-500/40 shadow-brand-500/5' : 'border-gray-800'));

    const avatarStyle = isAtMacro 
      ? 'border-emerald-500/70 text-emerald-400 bg-emerald-950/40' 
      : (isAtStructural ? 'border-amber-500/50 text-amber-400' : 'border-gray-700 text-white');

    html += `
      <div class="bg-dark-800 border ${cardBorder} rounded-2xl p-5 shadow-xl hover:border-gray-700 transition-all">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-dark-750 border ${avatarStyle} flex items-center justify-center font-bold text-sm">
              ${sym.slice(0, 3)}
            </div>
            <div>
              <div class="flex items-center gap-2">
                <h3 class="text-base font-bold text-white tracking-tight">${sym}</h3>
                <span class="text-xs text-gray-400 font-normal">(${sec})</span>
                <span class="px-2 py-0.5 rounded-full text-[10px] ${statusClass}">
                  ${statusText}
                </span>
                ${isAtMacro ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-black bg-emerald-500 text-black">⚡ ROCK-BOTTOM FLOOR</span>' : (isAtStructural ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-amber-500 text-black">★ STRUCTURAL FLOOR</span>' : (isPrime ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold badge-prime text-white">PRIME SETUP</span>' : ''))}
              </div>
              <p class="text-xs text-gray-400 mt-0.5">
                ${setupSubline}
              </p>
            </div>
          </div>

          <!-- Badges & Score -->
          <div class="flex items-center gap-2 flex-wrap">
            ${macroBadge}
            ${supportBadge}
            ${fvBadge}
            <div class="px-3 py-1 rounded-xl bg-dark-750 border border-gray-700 text-xs font-bold text-brand-400">
              Score: ${score}/100
            </div>
          </div>
        </div>

        <!-- Metric Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-4 pt-3 border-t border-gray-800/80 text-xs">
          <div>
            ${opp.status === "TARGET_HIT" ? `
              <span class="text-gray-400">Previous Target (Hit):</span>
              <p class="font-bold text-gray-400 text-sm mt-0.5 line-through">₹${target} <span class="text-[10px] font-bold text-brand-400 no-underline">HIT ✓</span></p>
              <p class="text-[11px] text-brand-400 mt-1 font-semibold">New Entry Target (+${targetPct}%): ₹${opp.new_entry_target || (price * (1 + targetPct/100)).toFixed(2)}</p>
            ` : `
              <span class="text-gray-400">Target (+${targetPct}%):</span>
              <p class="font-bold text-brand-400 text-sm mt-0.5">₹${target} <span class="text-[10px] font-normal text-gray-400">(${distTarget}% away)</span></p>
            `}
          </div>
          <div>
            <span class="text-gray-400">Rock-Bottom Floor:</span>
            <p class="font-bold text-emerald-400 text-sm mt-0.5">₹${opp.macro_floor_price || 'N/A'}</p>
            <p class="text-[10px] text-gray-400">+${opp.upside_to_cycle_high}% to peak</p>
          </div>
          <div>
            <span class="text-gray-400">3Y Range Position:</span>
            <p class="font-bold text-white text-sm mt-0.5">${opp.range_percentile}%</p>
            <p class="text-[10px] ${opp.range_percentile <= 25 ? 'text-emerald-400 font-semibold' : 'text-gray-400'}">${opp.range_percentile <= 25 ? 'Value Zone' : 'Mid/High Zone'}</p>
          </div>
          <div>
            <span class="text-gray-400">Volume Ratio:</span>
            <p class="font-semibold text-gray-200 mt-0.5">${opp.volume_ratio}x 20d avg</p>
            <p class="text-[10px] text-gray-400">Trend: <strong class="${opp.trend_regime === 'UPTREND' ? 'text-brand-400' : 'text-gray-300'}">${opp.trend_regime}</strong></p>
          </div>
          <div>
            <span class="text-gray-400">Deepest Dip (MAE):</span>
            <p class="font-semibold ${opp.mae_pct < 0 ? 'text-yellow-400' : 'text-gray-300'} mt-0.5">${opp.mae_pct}%</p>
            <p class="text-[10px] text-gray-400">No Stop Required</p>
          </div>
        </div>

        <!-- Rationale & Action Bar -->
        <div class="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-gray-800/60">
          <div class="flex items-center gap-2 flex-wrap text-xs text-gray-400">
            <i class="fa-solid fa-circle-check text-brand-400"></i>
            <span>${opp.reasons.join(" • ") || "Technical setup aligned"}</span>
          </div>

          <div class="flex items-center gap-2">
            <button onclick="quickInspectStock('${sym}')" class="px-3 py-1.5 rounded-lg bg-dark-750 hover:bg-dark-700 border border-gray-700 text-xs font-medium text-gray-300 transition-all">
              <i class="fa-solid fa-chart-line"></i> Inspect
            </button>
            <button onclick="quickTrackTrade('${sym}', ${price}, ${targetPct})" class="px-4 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 font-semibold text-white text-xs shadow-md shadow-brand-500/20 transition-all">
              <i class="fa-solid fa-shield-halved"></i> Track Trade (No-Stop)
            </button>
          </div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

// ---------------- TRACKER LOGIC ----------------
async function loadTrackerPositions(showFeedback = false) {
  try {
    const res = await fetch("/api/tracker/positions");
    const data = await res.json();

    document.getElementById("tr-total-trades").textContent = data.total_trades || 0;
    document.getElementById("tr-open-trades").textContent = data.open_trades_count || 0;
    document.getElementById("tr-hit-rate").textContent = (data.hit_rate_pct || 0) + "%";
    document.getElementById("tr-avg-days").textContent = (data.avg_sessions_to_hit || 0) + " d";

    const pill = document.getElementById("open-trades-pill");
    if (pill) pill.textContent = data.open_trades_count || 0;

    renderTrackerTable(data.positions || []);
  } catch (err) {
    console.error("Tracker load error:", err);
  }
}

async function refreshTracker() {
  const btn = document.getElementById("btn-refresh-tracker");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Refreshing...`;
  try {
    const res = await fetch("/api/tracker/refresh", { method: "POST" });
    const data = await res.json();
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> Refresh Outcomes`;
    loadTrackerPositions(false);
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> Refresh Outcomes`;
    alert("Refresh failed: " + err.message);
  }
}

function renderTrackerTable(positions) {
  const tbody = document.getElementById("tracker-table-body");
  if (!positions || positions.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="py-8 text-center text-gray-500">
          No active trades tracked yet. Scan for opportunities and click "Track Trade (No-Stop)" or add one manually.
        </td>
      </tr>
    `;
    return;
  }

  let html = "";
  positions.forEach(p => {
    const isHit = p.status === "TARGET_HIT";
    const pnl = p.unrealized_pnl_pct || 0;
    const pnlColor = pnl >= 0 ? "text-brand-400 font-bold" : "text-red-400 font-semibold";
    const statusBadge = isHit 
      ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-brand-500/20 text-brand-400 border border-brand-500/30">TARGET HIT (+${p.target_pct}%)</span>`
      : `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-500/20 text-purple-400 border border-purple-500/30">OPEN (${p.hold_sessions}d)</span>`;

    html += `
      <tr class="hover:bg-dark-750/50 transition-colors">
        <td class="py-3 px-4 font-bold text-white">${p.symbol}</td>
        <td class="py-3 px-4 text-gray-400">${p.entry_date}</td>
        <td class="py-3 px-4 font-medium text-gray-200">₹${p.entry_price}</td>
        <td class="py-3 px-4 font-semibold text-brand-400">₹${p.target_price} (+${p.target_pct}%)</td>
        <td class="py-3 px-4 font-medium text-white">₹${p.current_price || p.entry_price}</td>
        <td class="py-3 px-4 ${pnlColor}">${pnl >= 0 ? '+' : ''}${pnl}%</td>
        <td class="py-3 px-4 text-yellow-400 font-medium">${p.mae_pct || 0}%</td>
        <td class="py-3 px-4 text-gray-300 font-medium">${p.hold_sessions || 1} sessions</td>
        <td class="py-3 px-4">${statusBadge}</td>
        <td class="py-3 px-4 text-right">
          <button onclick="deletePosition('${p.id}')" class="text-gray-500 hover:text-red-400 transition-colors" title="Delete">
            <i class="fa-solid fa-trash-can"></i>
          </button>
        </td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

async function deletePosition(id) {
  if (!confirm("Remove this trade from your tracker?")) return;
  try {
    await fetch(`/api/tracker/${id}`, { method: "DELETE" });
    loadTrackerPositions(false);
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

// ---------------- BACKTEST LOGIC ----------------
function toggleBtMode() {
  const mode = document.getElementById("bt-mode").value;
  const scopeContainer = document.getElementById("bt-scope-container");
  const tickerContainer = document.getElementById("bt-ticker-container");
  if (mode === "basket") {
    scopeContainer.classList.remove("hidden");
    tickerContainer.classList.add("hidden");
  } else {
    scopeContainer.classList.add("hidden");
    tickerContainer.classList.remove("hidden");
  }
}

async function runBacktest() {
  const mode = document.getElementById("bt-mode").value;
  const targetPct = parseFloat(document.getElementById("bt-target-pct").value);
  const period = document.getElementById("bt-period").value;

  let payload = { mode, target_pct: targetPct, period, require_uptrend: false };

  if (mode === "basket") {
    payload.scope = parseInt(document.getElementById("bt-scope").value);
  } else {
    const ticker = document.getElementById("bt-ticker").value.trim().toUpperCase();
    if (!ticker) {
      alert("Please enter a ticker symbol");
      return;
    }
    payload.ticker = ticker;
  }

  const btn = document.getElementById("btn-run-bt");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Testing ${mode === "basket" ? "Universe" : payload.ticker}...`;

  try {
    const res = await fetch("/api/backtest/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText);
    }
    const data = await res.json();
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Simulation`;

    renderBacktestResults(data);
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Simulation`;
    alert("Backtest failed: " + err.message);
  }
}

function renderBacktestResults(data) {
  const panel = document.getElementById("bt-results-panel");
  panel.classList.remove("hidden");

  document.getElementById("bt-hit-rate").textContent = `${data.target_hit_rate_pct}%`;
  document.getElementById("bt-hits-count").textContent = data.target_hit_count;
  document.getElementById("bt-total-count").textContent = data.total_trades;

  document.getElementById("bt-avg-days").textContent = `${data.avg_sessions} d`;
  document.getElementById("bt-median-days").textContent = `${data.median_sessions}`;

  document.getElementById("bt-avg-mae").textContent = `${data.avg_mae_pct}%`;
  document.getElementById("bt-worst-mae").textContent = `${data.worst_mae_pct}%`;

  // Duration bars
  const dist = data.distribution || {};
  const total = data.target_hit_count || 1;
  const barsContainer = document.getElementById("bt-duration-bars");

  const bins = [
    { label: "1 to 5 Days", key: "1_to_5_days", color: "bg-brand-500" },
    { label: "6 to 15 Days", key: "6_to_15_days", color: "bg-blue-500" },
    { label: "16 to 30 Days", key: "16_to_30_days", color: "bg-purple-500" },
    { label: "31 to 60 Days", key: "31_to_60_days", color: "bg-yellow-500" },
    { label: "> 60 Days", key: "over_60_days", color: "bg-red-500" },
  ];

  let barsHtml = "";
  bins.forEach(b => {
    const count = dist[b.key] || 0;
    const pct = Math.round((count / total) * 100);
    barsHtml += `
      <div class="bg-dark-750 p-3 rounded-xl border border-gray-700/60">
        <p class="text-xs text-gray-400 font-medium">${b.label}</p>
        <p class="text-lg font-bold text-white mt-1">${count} <span class="text-xs font-normal text-gray-500">(${pct}%)</span></p>
        <div class="w-full bg-gray-700 h-1.5 rounded-full mt-2 overflow-hidden">
          <div class="${b.color} h-full rounded-full" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  });
  barsContainer.innerHTML = barsHtml;

  // Per-Ticker Performance Breakdown (For Multi-Ticker Baskets)
  const summaryContainer = document.getElementById("bt-ticker-summary-container");
  if (data.is_basket && data.ticker_summaries && data.ticker_summaries.length > 0) {
    summaryContainer.classList.remove("hidden");
    document.getElementById("bt-tested-count-badge").textContent = `${data.tickers_tested} stocks analyzed`;
    const sumTbody = document.getElementById("bt-summary-table-body");
    let sumHtml = "";
    data.ticker_summaries.forEach(s => {
      const hitClass = s.hit_rate_pct >= 90 ? "text-brand-400 font-bold" : "text-yellow-400 font-medium";
      sumHtml += `
        <tr class="hover:bg-dark-750/50">
          <td class="py-2.5 px-4 font-bold text-white">${s.symbol}</td>
          <td class="py-2.5 px-4 text-gray-400">${s.security}</td>
          <td class="py-2.5 px-4">${s.total_trades}</td>
          <td class="py-2.5 px-4 text-brand-400 font-semibold">${s.target_hit_count}</td>
          <td class="py-2.5 px-4 ${hitClass}">${s.hit_rate_pct}%</td>
          <td class="py-2.5 px-4">${s.avg_sessions} d</td>
          <td class="py-2.5 px-4 text-yellow-400">${s.worst_mae_pct}%</td>
        </tr>
      `;
    });
    sumTbody.innerHTML = sumHtml;
  } else {
    summaryContainer.classList.add("hidden");
  }

  // Trades Replay Table
  const tbody = document.getElementById("bt-table-body");
  const trades = data.trades || [];
  const logCountEl = document.getElementById("bt-log-count-text");
  if (logCountEl) logCountEl.textContent = `${trades.length} trades recorded across history`;

  if (trades.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="py-4 text-center text-gray-500">No trades triggered in this period.</td></tr>`;
    return;
  }

  let tableHtml = "";
  trades.forEach(t => {
    const isHit = t.status === "TARGET_HIT";
    const statusText = isHit 
      ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-brand-500/20 text-brand-400 border border-brand-500/30">TARGET HIT</span>` 
      : `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-500/20 text-purple-400 border border-purple-500/30">OPEN</span>`;
    const sym = t.symbol || data.ticker || "STOCK";
    tableHtml += `
      <tr class="hover:bg-dark-750/50">
        <td class="py-2.5 px-4 font-bold text-white">${sym}</td>
        <td class="py-2.5 px-4 text-gray-400">${t.entry_date}</td>
        <td class="py-2.5 px-4 font-medium text-gray-200">₹${t.entry_price}</td>
        <td class="py-2.5 px-4 font-semibold text-brand-400">₹${t.target_price}</td>
        <td class="py-2.5 px-4 text-gray-400">${t.exit_date || 'Still Open'}</td>
        <td class="py-2.5 px-4 font-medium">${t.hold_sessions} days</td>
        <td class="py-2.5 px-4 text-yellow-400 font-medium">${t.mae_pct}% (₹${t.lowest_price})</td>
        <td class="py-2.5 px-4">${statusText}</td>
      </tr>
    `;
  });
  tbody.innerHTML = tableHtml;
}

// ---------------- VALUATION & SUPPORT INSPECTOR ----------------
async function inspectStockValuation() {
  const ticker = document.getElementById("fv-search-ticker").value.trim().toUpperCase();
  if (!ticker) return;

  try {
    const res = await fetch(`/api/stock/${ticker}`);
    if (!res.ok) {
      const errText = await res.text();
      let msg = errText;
      try { msg = JSON.parse(errText).detail || errText; } catch (_) {}
      alert(`Failed to load ${ticker}: ${msg}`);
      return;
    }
    const data = await res.json();
    renderStockDetails(data);
  } catch (err) {
    alert("Failed to load stock data: " + err.message);
  }
}

function quickInspectStock(sym) {
  document.getElementById("fv-search-ticker").value = sym;
  switchTab("tab-valuation");
  inspectStockValuation();
}

function renderStockDetails(data) {
  const panel = document.getElementById("fv-results-panel");
  panel.classList.remove("hidden");

  const fv = data.fair_value || {};
  const support = data.support;

  document.getElementById("fv-current-price").textContent = `₹${fv.current_price || '0.00'}`;
  document.getElementById("fv-as-of").textContent = `As of ${data.as_of || 'Latest session'}`;

  document.getElementById("fv-composite-price").textContent = `₹${fv.composite_fair_price || 'N/A'}`;
  document.getElementById("fv-bucket").textContent = fv.valuation_bucket || 'N/A';

  const gapEl = document.getElementById("fv-gap-pct");
  if (fv.fair_gap_pct !== null && fv.fair_gap_pct !== undefined) {
    gapEl.textContent = `${fv.fair_gap_pct >= 0 ? '+' : ''}${fv.fair_gap_pct}%`;
    gapEl.className = fv.fair_gap_pct >= 0 ? "text-3xl font-extrabold text-brand-400 mt-2" : "text-3xl font-extrabold text-yellow-400 mt-2";
  } else {
    gapEl.textContent = "N/A";
  }

  // 0. Macro Range Channel & Powerful Rock-Bottom Floor
  const mf = data.macro_floor || (data.opportunity ? {
    macro_floor_price: data.opportunity.macro_floor_price,
    touches: data.opportunity.macro_touches,
    touch_dates: data.opportunity.macro_dates,
    max_historical_rally_pct: data.opportunity.macro_rally_pct,
    prox_to_macro_floor_pct: data.opportunity.macro_proximity_pct,
    range_percentile: data.opportunity.range_percentile,
    cycle_low: data.opportunity.cycle_low,
    cycle_high: data.opportunity.cycle_high,
    upside_to_cycle_high: data.opportunity.upside_to_cycle_high,
    gain_probability_rating: data.opportunity.gain_probability_rating,
    is_at_macro_floor: data.opportunity.is_at_macro_floor,
    is_rebounding_from_macro: data.opportunity.is_rebounding_from_macro,
  } : null);

  if (mf) {
    if (document.getElementById("fv-cycle-low")) document.getElementById("fv-cycle-low").textContent = `₹${mf.cycle_low}`;
    if (document.getElementById("fv-cycle-high")) document.getElementById("fv-cycle-high").textContent = `₹${mf.cycle_high}`;
    if (document.getElementById("fv-range-pct")) document.getElementById("fv-range-pct").textContent = `${mf.range_percentile}% of 3Y Cycle`;
    if (document.getElementById("fv-range-fill")) document.getElementById("fv-range-fill").style.width = `${Math.min(Math.max(mf.range_percentile, 5), 100)}%`;
    if (document.getElementById("fv-upside-to-high")) document.getElementById("fv-upside-to-high").textContent = `+${mf.upside_to_cycle_high}% upside to 3-year peak (₹${mf.cycle_high})`;
    
    const probBadge = document.getElementById("fv-gain-prob-badge");
    if (probBadge) {
      probBadge.textContent = mf.gain_probability_rating || "MODERATE";
      if (mf.gain_probability_rating && mf.gain_probability_rating.startsWith("VERY HIGH")) {
        probBadge.className = "px-3 py-1 rounded-full text-xs font-black bg-emerald-500 text-black shadow-lg shadow-emerald-500/30";
      } else if (mf.gain_probability_rating && mf.gain_probability_rating.startsWith("HIGH")) {
        probBadge.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50";
      } else {
        probBadge.className = "px-3 py-1 rounded-full text-xs font-medium bg-gray-700 text-gray-300";
      }
    }

    if (document.getElementById("fv-macro-price")) document.getElementById("fv-macro-price").textContent = `₹${mf.macro_floor_price}`;
    if (document.getElementById("fv-macro-prox")) document.getElementById("fv-macro-prox").textContent = `${mf.prox_to_macro_floor_pct > 0 ? '+' : ''}${mf.prox_to_macro_floor_pct}%`;
    if (document.getElementById("fv-macro-rally")) document.getElementById("fv-macro-rally").textContent = `+${mf.max_historical_rally_pct}%`;
    if (document.getElementById("fv-macro-touches")) document.getElementById("fv-macro-touches").textContent = `${mf.touches} Cycle Tests`;
    if (document.getElementById("fv-macro-dates")) document.getElementById("fv-macro-dates").innerHTML = `<span class="font-bold text-emerald-400">Institutional Accumulation Dates:</span> ${(mf.touch_dates || []).join(", ")} • <span class="text-brand-400 font-semibold">Subsequent Expansion: +${mf.max_historical_rally_pct}%</span>`;
    
    const macroBadge = document.getElementById("fv-macro-status-badge");
    if (macroBadge) {
      if (mf.is_at_macro_floor) {
        macroBadge.textContent = "★ ACTIVE ROCK-BOTTOM FLOOR TEST";
        macroBadge.className = "px-3 py-1 rounded-full text-xs font-black bg-emerald-500 text-black shadow-md shadow-emerald-500/30";
      } else if (mf.is_rebounding_from_macro) {
        macroBadge.textContent = "REBOUNDING OFF ROCK-BOTTOM FLOOR";
        macroBadge.className = "px-3 py-1 rounded-full text-xs font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-500/50";
      } else {
        macroBadge.textContent = `Historical Macro Floor (₹${mf.macro_floor_price})`;
        macroBadge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-dark-750 text-gray-300 border border-gray-700";
      }
    }
  }

  // 1. PRIME FOCUS: Structural Low Support Floor Card
  const structCard = document.getElementById("fv-structural-card");
  const st = data.structural_support;
  if (st && structCard) {
    structCard.classList.remove("hidden");
    document.getElementById("fv-structural-price").textContent = `₹${st.structural_low_price}`;
    document.getElementById("fv-structural-prox").textContent = `${st.structural_proximity_pct > 0 ? '+' : ''}${st.structural_proximity_pct}%`;
    document.getElementById("fv-structural-status-sub").textContent = st.structural_status === "AT_STRUCTURAL_LOW" ? "Active Macro Accumulation Floor" : "Rebounding off Structural Floor";
    document.getElementById("fv-structural-rally").textContent = `+${st.max_historical_rally_pct}%`;
    document.getElementById("fv-structural-touches").textContent = `${st.structural_touches} Cycle Tests`;
    document.getElementById("fv-structural-dates").innerHTML = `<span class="font-bold text-amber-400">Defended Turning Points:</span> ${st.structural_dates.join(", ")} • <span class="text-brand-400 font-semibold">Subsequent Surge: +${st.max_historical_rally_pct}%</span>`;
  } else if (structCard) {
    document.getElementById("fv-structural-price").textContent = "None Active";
    document.getElementById("fv-structural-prox").textContent = "N/A";
    document.getElementById("fv-structural-rally").textContent = "N/A";
    document.getElementById("fv-structural-touches").textContent = "0";
    document.getElementById("fv-structural-dates").textContent = "Current price is not within ±6% of an active macro structural low.";
  }

  // 2. Horizontal Support Card
  const supportCard = document.getElementById("fv-support-card");
  if (support) {
    supportCard.classList.remove("hidden");
    document.getElementById("fv-support-price").textContent = `₹${support.support_level}`;
    document.getElementById("fv-support-touches").textContent = `${support.touches} Touches`;
    document.getElementById("fv-support-prox").textContent = `${support.proximity_pct > 0 ? '+' : ''}${support.proximity_pct}% (${support.status})`;
    document.getElementById("fv-support-dates").textContent = `Historical Bounces: ${support.touch_dates.join(", ")}`;
  } else {
    document.getElementById("fv-support-price").textContent = "None Active";
    document.getElementById("fv-support-touches").textContent = "0";
    document.getElementById("fv-support-prox").textContent = "N/A";
    document.getElementById("fv-support-dates").textContent = "No active horizontal floor within -2% to +5.5% of current price.";
  }

  // 3. All historical floors table (Structural + Horizontal)
  const allFloorsTable = document.getElementById("fv-all-floors-tbody");
  const allFloorsCount = document.getElementById("fv-all-floors-count");
  const allFloors = data.all_structural_floors || data.all_floors || [];

  if (allFloorsCount) {
    allFloorsCount.textContent = `${allFloors.length} Floors Detected`;
  }

  if (allFloorsTable) {
    if (allFloors.length === 0) {
      allFloorsTable.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-gray-500">No multi-touch structural floors detected.</td></tr>`;
    } else {
      let fHtml = "";
      allFloors.forEach(f => {
        const isActive = f.is_active_now;
        const priceVal = f.structural_low_price || f.support_level;
        const statusBadge = isActive 
          ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500 text-black shadow-sm shadow-amber-500/30">★ ACTIVE NOW</span>`
          : (f.status === "ABOVE_FLOOR" || f.status === "ABOVE_STRUCTURAL_LOW" 
              ? `<span class="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-700 text-gray-300">HISTORICAL FLOOR</span>` 
              : `<span class="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-800 text-gray-400">${f.status}</span>`);

        const proxColor = isActive ? "text-amber-400 font-bold" : "text-gray-400";
        const proxSign = f.proximity_pct > 0 ? "+" : "";
        const rallyInfo = f.max_historical_rally_pct ? ` (Impulse: +${f.max_historical_rally_pct}%)` : '';

        fHtml += `
          <tr class="hover:bg-dark-750/50 transition-colors">
            <td class="py-2.5 px-4 font-extrabold text-white text-sm">₹${priceVal}</td>
            <td class="py-2.5 px-4"><span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold">${f.touches} tests${rallyInfo}</span></td>
            <td class="py-2.5 px-4 ${proxColor}">${proxSign}${f.proximity_pct}%</td>
            <td class="py-2.5 px-4">${statusBadge}</td>
            <td class="py-2.5 px-4 text-gray-400 text-[11px]">${(f.touch_dates || []).slice(-6).join(", ")}</td>
          </tr>
        `;
      });
      allFloorsTable.innerHTML = fHtml;
    }
  }

  // Fundamental checks
  const checksList = document.getElementById("fv-checks-list");
  const checks = fv.checks || [];
  if (checks.length > 0) {
    let checksHtml = "";
    checks.forEach(c => {
      const icon = c.passed ? '<i class="fa-solid fa-circle-check text-brand-400"></i>' : '<i class="fa-solid fa-circle-xmark text-red-400"></i>';
      checksHtml += `
        <div class="bg-dark-750 p-3 rounded-xl border border-gray-700/60 flex items-center justify-between">
          <div>
            <p class="text-xs font-semibold text-white">${c.name}</p>
            <p class="text-[10px] text-gray-400 mt-0.5">${c.detail}</p>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs font-bold text-gray-200">${c.value}</span>
            ${icon}
          </div>
        </div>
      `;
    });
    checksList.innerHTML = checksHtml;
  } else {
    checksList.innerHTML = `<p class="text-xs text-gray-500 col-span-2">Pre-computed Strict-4 fundamental checks unavailable for this ticker.</p>`;
  }
}

// ---------------- MODAL LOGIC ----------------
function quickTrackTrade(sym, price, targetPct) {
  document.getElementById("modal-sym").value = sym;
  document.getElementById("modal-price").value = price;
  document.getElementById("modal-target-pct").value = targetPct || 3.0;
  document.getElementById("modal-notes").value = "Scanned 9EMA confirmation setup";
  document.getElementById("modal-add-trade").classList.remove("hidden");
}

function openManualTradeModal() {
  document.getElementById("modal-sym").value = "";
  document.getElementById("modal-price").value = "";
  document.getElementById("modal-target-pct").value = "3.0";
  document.getElementById("modal-notes").value = "";
  document.getElementById("modal-add-trade").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal-add-trade").classList.add("hidden");
}

async function submitTrackTrade() {
  const symbol = document.getElementById("modal-sym").value.trim().toUpperCase();
  const entry_price = parseFloat(document.getElementById("modal-price").value);
  const target_pct = parseFloat(document.getElementById("modal-target-pct").value);
  const capital = parseFloat(document.getElementById("modal-capital").value);
  const notes = document.getElementById("modal-notes").value;

  if (!symbol || !entry_price || entry_price <= 0) {
    alert("Please provide a valid symbol and entry price.");
    return;
  }

  try {
    const res = await fetch("/api/tracker/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, entry_price, target_pct, capital, notes })
    });
    if (!res.ok) throw new Error("Failed to save trade");
    
    closeModal();
    switchTab("tab-tracker");
  } catch (err) {
    alert("Failed to track trade: " + err.message);
  }
}
