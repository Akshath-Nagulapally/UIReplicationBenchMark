const app = document.querySelector("#app");
const REFRESH_INTERVAL_MS = 5000;

renderShell();
refreshResults();
window.setInterval(refreshResults, REFRESH_INTERVAL_MS);

async function refreshResults() {
  setStatus("Refreshing benchmark results...");

  try {
    const response = await fetch("/api/results", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    const payload = await response.json();
    renderResults(payload);
    setStatus(`Watching ${payload.runsDir} for updates`);
  } catch (error) {
    renderError(error);
    setStatus("Could not load results");
  }
}

function renderShell() {
  app.innerHTML = `
    <div class="page">
      <header class="hero">
        <div>
          <p class="eyebrow">Visualize</p>
          <h1>UI Replication Benchmark</h1>
          <p class="lede">
            Live comparison for target and generated screenshots with auto-refreshing scores.
          </p>
        </div>
      </header>
      <section class="toolbar">
        <div id="status" class="status">Starting…</div>
        <div id="timestamp" class="timestamp"></div>
      </section>
      <main id="content" class="content"></main>
    </div>
  `;
}

function renderResults(payload) {
  const content = document.querySelector("#content");
  const timestamp = document.querySelector("#timestamp");
  timestamp.textContent = `Last refresh ${formatTimestamp(payload.generatedAt)}`;

  if (!payload.results.length) {
    content.innerHTML = `
      <section class="empty-state">
        <h2>No runs found yet</h2>
        <p>Add screenshots under <code>${escapeHtml(payload.runsDir)}/*</code> to populate the leaderboard.</p>
      </section>
    `;
    return;
  }

  const metricHeaders = payload.scoreNames
    .map((metricName) => `<th>${escapeHtml(metricName)} Score</th>`)
    .join("");

  const rows = payload.results
    .map((result, index) => {
      const metricCells = payload.scoreNames
        .map((metricName) => renderScoreCell(result.scores[metricName]))
        .join("");

      return `
        <tr>
          <td class="number">${escapeHtml(String(index + 1))}</td>
          <td>
            <strong>${escapeHtml(result.name)}</strong>
          </td>
          <td>${renderImageCell(result.targetImageUrl, `Target for ${result.name}`)}</td>
          <td>${renderImageCell(result.candidateImageUrl, `Candidate for ${result.name}`)}</td>
          ${metricCells}
        </tr>
      `;
    })
    .join("");

  content.innerHTML = `
    <section class="results-panel">
      <div class="panel-header">
        <div>
          <h2>Run Leaderboard</h2>
          <p>${payload.results.length} runs sorted by similarity score. Higher is better: <code>1</code> is a near-exact match and <code>0</code> is unrelated or reward-hacked.</p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Run</th>
              <th>Target</th>
              <th>Candidate</th>
              ${metricHeaders}
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderImageCell(src, alt) {
  return `
    <div class="image-frame">
      <img src="${escapeAttribute(src)}" alt="${escapeAttribute(alt)}" loading="lazy" />
    </div>
  `;
}

function renderScoreCell(score) {
  if (score && typeof score === "object" && !Array.isArray(score)) {
    if (score.request_success === false) {
      const reason = score.reason ? ` title="${escapeAttribute(score.reason)}"` : "";
      return `<td><span class="score-error"${reason}>request failed</span></td>`;
    }

    if (typeof score.value === "number" && Number.isFinite(score.value)) {
      const detailParts = [];
      if (typeof score.raw_similarity === "number" && Number.isFinite(score.raw_similarity)) {
        detailParts.push(`raw ${score.raw_similarity.toFixed(3)}`);
      }
      if (score.reward_hacking === true) {
        detailParts.push("reward hacking detected");
      }
      if (typeof score.reason === "string" && score.reason) {
        detailParts.push(score.reason);
      }
      const title = detailParts.length ? ` title="${escapeAttribute(detailParts.join(" | "))}"` : "";
      return `<td class="number"${title}>${score.value.toFixed(6)}</td>`;
    }
  }

  if (typeof score === "number" && Number.isFinite(score)) {
    return `<td class="number">${score.toFixed(6)}</td>`;
  }

  return `<td><span class="score-error">${escapeHtml(String(score))}</span></td>`;
}

function renderNumeric(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(3);
  }
  return "—";
}

function renderError(error) {
  const content = document.querySelector("#content");
  content.innerHTML = `
    <section class="empty-state error-state">
      <h2>Could not load results</h2>
      <p>${escapeHtml(error.message || String(error))}</p>
    </section>
  `;
}

function setStatus(text) {
  const status = document.querySelector("#status");
  status.textContent = text;
}

function formatTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}
