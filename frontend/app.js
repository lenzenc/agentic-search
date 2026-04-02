// ── State ────────────────────────────────────────────────────────────────────
let isSearching = false;
let startTime = null;
let rawLogLines = [];

// ── DOM refs ─────────────────────────────────────────────────────────────────
const searchInput  = document.getElementById("search-input");
const searchBtn    = document.getElementById("search-btn");
const traceSteps   = document.getElementById("trace-steps");
const traceDot     = document.getElementById("trace-dot");
const answerText   = document.getElementById("answer-text");
const cardsGrid    = document.getElementById("cards-grid");
const patternBadge = document.getElementById("pattern-badge");
const rawLogEl     = document.getElementById("raw-log");
const statsBar     = document.getElementById("stats-bar");

// ── Node metadata ─────────────────────────────────────────────────────────────
const NODE_META = {
  analyze_query:  { icon: "🔍", colorClass: "analyze",  label: "Analyze Query" },
  retrieve:       { icon: "📡", colorClass: "retrieve", label: "Retrieve" },
  grade_documents:{ icon: "⚖️", colorClass: "grade",    label: "Grade Documents" },
  rewrite_query:  { icon: "↺",  colorClass: "rewrite",  label: "Rewrite Query" },
  generate_answer:{ icon: "✨", colorClass: "generate", label: "Generate Answer" },
};

// ── Example queries ───────────────────────────────────────────────────────────
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    searchInput.value = btn.dataset.query;
    doSearch();
  });
});

// ── Search trigger ────────────────────────────────────────────────────────────
searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

// ── Raw log toggle ────────────────────────────────────────────────────────────
document.getElementById("raw-log-toggle").addEventListener("click", () => {
  rawLogEl.classList.toggle("visible");
  document.getElementById("raw-log-toggle-text").textContent =
    rawLogEl.classList.contains("visible") ? "Hide raw log" : "Show raw log";
});

// ── Main search function ──────────────────────────────────────────────────────
async function doSearch() {
  const query = searchInput.value.trim();
  if (!query || isSearching) return;

  isSearching = true;
  startTime = Date.now();
  rawLogLines = [];

  // Reset UI
  searchBtn.disabled = true;
  searchBtn.textContent = "Searching…";
  traceSteps.innerHTML = "";
  rawLogEl.innerHTML = "";
  answerText.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
  cardsGrid.innerHTML = "";
  patternBadge.className = "pattern-badge";
  patternBadge.textContent = "";
  statsBar.style.display = "none";
  traceDot.classList.add("active");

  appendLog("INFO", "search", `started query="${query}"`);

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Search failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const messages = buffer.split("\n\n");
      buffer = messages.pop(); // last may be incomplete

      for (const msg of messages) {
        if (!msg.trim()) continue;
        const lines = msg.split("\n");
        let eventType = "message";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7).trim();
          else if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
        }
        if (!dataStr) continue;

        try {
          const data = JSON.parse(dataStr);
          handleEvent(eventType, data);
        } catch {
          // ignore parse errors
        }
      }
    }
  } catch (err) {
    answerText.innerHTML = `<span style="color:var(--red)">Error: ${escHtml(err.message)}</span>`;
  } finally {
    isSearching = false;
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
    traceDot.classList.remove("active");
    // Mark last step as no longer active
    document.querySelectorAll(".trace-step.active").forEach(el => el.classList.remove("active"));
  }
}

// ── SSE event handler ─────────────────────────────────────────────────────────
function handleEvent(eventType, data) {
  if (eventType === "trajectory_step") {
    renderTrajectoryStep(data);
    appendLog("INFO", data.node, data.detail);
  } else if (eventType === "result") {
    renderResult(data);
    appendLog("INFO", "result", `pattern=${data.pattern} iterations=${data.total_iterations} cards=${data.cards.length}`);
  } else if (eventType === "error") {
    answerText.innerHTML = `<span style="color:var(--red)">Error: ${escHtml(data.error)}</span>`;
    appendLog("ERROR", "system", data.error);
  }
}

// ── Trajectory step renderer ──────────────────────────────────────────────────
function renderTrajectoryStep(step) {
  const meta = NODE_META[step.node] || { icon: "●", colorClass: "", label: step.node };
  const elapsed = startTime ? `${((Date.now() - startTime) / 1000).toFixed(1)}s` : "";

  // Mark previous steps as no longer active
  document.querySelectorAll(".trace-step.active").forEach(el => el.classList.remove("active"));

  const el = document.createElement("div");
  el.className = "trace-step active";
  el.dataset.node = step.node;

  // Build badges
  let badges = "";
  if (step.node === "rewrite_query") {
    const iter = step.metadata?.iteration || "";
    const max  = step.metadata?.max_iterations || "";
    badges = `<span class="loop-badge">↺ loop ${iter} of ${max}</span>`;
  }
  if (step.node === "analyze_query" && step.metadata?.is_complex) {
    const n = (step.metadata?.sub_queries || []).length;
    badges = `<span class="expand-badge">⇥ expanded to ${n} sub-queries</span>`;
  }

  el.innerHTML = `
    <div class="trace-step-header" onclick="toggleStep(this.parentElement)">
      <span class="node-icon">${meta.icon}</span>
      <span class="node-name ${meta.colorClass}">${meta.label}</span>
      ${badges}
      <span class="step-time">${elapsed}</span>
      <span class="step-chevron">▶</span>
    </div>
    <div class="trace-step-body">
      <div class="step-detail">${escHtml(step.detail)}</div>
      ${buildStepBody(step)}
    </div>`;

  traceSteps.appendChild(el);
  traceSteps.scrollTop = traceSteps.scrollHeight;

  // Auto-open current step
  el.classList.add("open");
}

function buildStepBody(step) {
  const m = step.metadata || {};
  const parts = [];

  if (step.node === "analyze_query" && m.is_complex && m.sub_queries?.length) {
    parts.push(`<div class="subquery-list">${
      m.sub_queries.map(q => `<div class="subquery-item">${escHtml(q)}</div>`).join("")
    }</div>`);
  }

  if (step.node === "retrieve" && m.top_cards?.length) {
    parts.push(`<div class="retrieved-list">${
      m.top_cards.map(n => `<span class="card-chip">${escHtml(n)}</span>`).join("")
    }</div>`);
    if (m.queries?.length > 1) {
      parts.push(`<div style="margin-top:6px;font-size:11px">Searched ${m.queries.length} sub-queries</div>`);
    }
  }

  if (step.node === "grade_documents" && m.grades?.length) {
    parts.push(`<div class="grade-list">${
      m.grades.map(g => `
        <div class="grade-item ${g.grade === "relevant" ? "relevant" : "not-relevant"}">
          <span class="grade-icon">${g.grade === "relevant" ? "✓" : "✗"}</span>
          <div>
            <div class="grade-name">${escHtml(g.name)}</div>
            <div class="grade-reason">${escHtml(g.reasoning)}</div>
          </div>
        </div>`).join("")
    }</div>`);
  }

  if (step.node === "rewrite_query") {
    parts.push(`
      <div style="margin-top:6px;font-size:12px">
        <div><span style="color:var(--text-muted)">Previous query:</span> <span style="color:var(--red)">${escHtml(m.old_query || "")}</span></div>
        <div style="margin-top:4px"><span style="color:var(--text-muted)">New query:</span> <span style="color:var(--green)">${escHtml(m.new_query || "")}</span></div>
      </div>`);
  }

  if (step.node === "generate_answer" && m.card_names?.length) {
    parts.push(`<div style="margin-top:6px;font-size:11px;color:var(--text-muted)">
      Cards: ${m.card_names.map(n => escHtml(n)).join(", ")}
    </div>`);
  }

  return parts.join("");
}

// ── Final result renderer ─────────────────────────────────────────────────────
function renderResult(data) {
  // Answer — render as markdown
  answerText.innerHTML = data.answer
    ? marked.parse(data.answer)
    : "No answer generated.";

  // Pattern badge
  if (data.pattern === "self_correcting_loop") {
    patternBadge.className = "pattern-badge self-correcting";
    patternBadge.textContent = "↺ Self-Correcting Loop";
  } else if (data.pattern === "query_expansion") {
    patternBadge.className = "pattern-badge query-expansion";
    patternBadge.textContent = "⇥ Query Expansion";
  } else {
    patternBadge.className = "pattern-badge direct";
    patternBadge.textContent = "Direct";
  }

  // Cards grid
  cardsGrid.innerHTML = "";
  if (data.cards?.length) {
    data.cards.forEach(card => {
      const tile = document.createElement("div");
      tile.className = "card-tile";
      const imgSrc = card.image_small || card.image_large || "";
      const typeClass = `type-${(card.types?.[0] || "colorless").toLowerCase()}`;
      tile.innerHTML = `
        ${imgSrc ? `<img src="${escAttr(imgSrc)}" alt="${escAttr(card.name)}" loading="lazy">` : `<div style="aspect-ratio:2.5/3.5;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:32px">🃏</div>`}
        <div class="card-tile-info">
          <div class="card-tile-name">${escHtml(card.name)}</div>
          <div class="card-tile-meta ${typeClass}">${card.types?.join("/") || card.supertype || ""}</div>
          <div class="card-tile-meta">HP ${card.hp ?? "—"} · ${card.stage || card.supertype || ""}</div>
        </div>`;
      cardsGrid.appendChild(tile);
    });
  } else {
    cardsGrid.innerHTML = `<div class="answer-placeholder">No cards to display.</div>`;
  }

  // Stats bar
  const elapsed = startTime ? `${((Date.now() - startTime) / 1000).toFixed(1)}s` : "—";
  statsBar.innerHTML = `
    <div class="stat"><span>Total time:</span><span class="stat-val">${elapsed}</span></div>
    <div class="stat"><span>Iterations:</span><span class="stat-val">${data.total_iterations}</span></div>
    <div class="stat"><span>Cards found:</span><span class="stat-val">${data.cards?.length || 0}</span></div>
    <div class="stat"><span>Pattern:</span><span class="stat-val">${data.pattern?.replace(/_/g, " ") || "—"}</span></div>`;
  statsBar.style.display = "flex";
}

// ── Raw log ───────────────────────────────────────────────────────────────────
function appendLog(level, node, msg) {
  const now = new Date();
  const ts = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}.${String(now.getMilliseconds()).padStart(3,"0")}`;
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="ts">[${ts}]</span> <span class="level">${level.padEnd(5)}</span> <span class="node-tag">${node.padEnd(18)}</span> ${escHtml(msg)}`;
  rawLogEl.appendChild(line);
  rawLogEl.scrollTop = rawLogEl.scrollHeight;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function toggleStep(el) {
  el.classList.toggle("open");
}

function escHtml(s) {
  if (!s) return "";
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function escAttr(s) {
  if (!s) return "";
  return String(s).replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
