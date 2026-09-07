// Go Board Canvas & API Controller

const API_BASE = window.location.origin;
const canvas = document.getElementById("goBoard");
const ctx = canvas.getContext("2d");

let boardSize = 19;
let grid = Array(boardSize).fill(0).map(() => Array(boardSize).fill(0));
let toMove = "B";
let lastMove = null;
let topCandidates = [];

const GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

function getStarPoints(size) {
  if (size === 19) {
    return [[3, 3], [3, 9], [3, 15], [9, 3], [9, 9], [9, 15], [15, 3], [15, 9], [15, 15]];
  } else if (size === 13) {
    return [[3, 3], [3, 9], [6, 6], [9, 3], [9, 9]];
  } else if (size === 9) {
    return [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];
  }
  return [];
}

function drawBoard() {
  const width = canvas.width;
  const padding = 25;
  const cell = (width - 2 * padding) / (boardSize - 1);

  ctx.clearRect(0, 0, width, width);

  // Background
  ctx.fillStyle = "#e2b069";
  ctx.fillRect(0, 0, width, width);

  // Grid Lines
  ctx.strokeStyle = "#5c3a10";
  ctx.lineWidth = 1.2;
  for (let i = 0; i < boardSize; i++) {
    // Horizontal
    ctx.beginPath();
    ctx.moveTo(padding, padding + i * cell);
    ctx.lineTo(width - padding, padding + i * cell);
    ctx.stroke();

    // Vertical
    ctx.beginPath();
    ctx.moveTo(padding + i * cell, padding);
    ctx.lineTo(padding + i * cell, width - padding);
    ctx.stroke();
  }

  // Star points
  const stars = getStarPoints(boardSize);
  ctx.fillStyle = "#5c3a10";
  for (const [r, c] of stars) {
    ctx.beginPath();
    ctx.arc(padding + c * cell, padding + r * cell, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw Stones
  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      const stone = grid[r][c];
      if (stone === 1) {
        drawStone(r, c, "black", padding, cell);
      } else if (stone === 2) {
        drawStone(r, c, "white", padding, cell);
      }
    }
  }

  // Highlight Last Move
  if (lastMove && lastMove.r >= 0 && lastMove.c >= 0) {
    const x = padding + lastMove.c * cell;
    const y = padding + lastMove.r * cell;
    ctx.strokeStyle = lastMove.color === "B" ? "#ffffff" : "#000000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, cell * 0.2, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Draw Candidate Move Overlays
  topCandidates.forEach((cand, idx) => {
    if (cand.move && cand.move[0] >= 0) {
      const [cr, cc] = cand.move;
      const cx = padding + cc * cell;
      const cy = padding + cr * cell;

      ctx.fillStyle = `rgba(56, 189, 248, ${0.85 - idx * 0.12})`;
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.38, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#0f172a";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${idx + 1}`, cx, cy);
    }
  });
}

function drawStone(r, c, type, padding, cell) {
  const x = padding + c * cell;
  const y = padding + r * cell;
  const radius = cell * 0.46;

  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);

  const grad = ctx.createRadialGradient(
    x - radius * 0.3,
    y - radius * 0.3,
    radius * 0.1,
    x,
    y,
    radius
  );

  if (type === "black") {
    grad.addColorStop(0, "#475569");
    grad.addColorStop(1, "#090d16");
  } else {
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(1, "#cbd5e1");
  }

  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.4)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

async function fetchState() {
  try {
    const res = await fetch(`${API_BASE}/api/board/state`);
    const data = await res.json();
    boardSize = data.board_size;
    grid = data.grid;
    toMove = data.to_move;

    document.getElementById("turnDisplay").textContent =
      toMove === "B" ? "Black (⚫)" : "White (⚪)";
    document.getElementById("blackCaptures").textContent = data.captures.black;
    document.getElementById("whiteCaptures").textContent = data.captures.white;
    document.getElementById("scoreDisplay").textContent = data.score;

    drawBoard();
  } catch (err) {
    console.error("Fetch state error:", err);
  }
}

async function playCoord(r, c) {
  const colChar = GTP_COLUMNS[c];
  const rowNum = boardSize - r;
  const gtp = `${colChar}${rowNum}`;

  try {
    const res = await fetch(`${API_BASE}/api/board/play`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ color: toMove, coord: gtp }),
    });
    if (res.ok) {
      lastMove = { r, c, color: toMove };
      topCandidates = [];
      await fetchState();
    }
  } catch (err) {
    console.error("Play error:", err);
  }
}

async function requestBotMove() {
  try {
    const res = await fetch(`${API_BASE}/api/optimize/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_simulations: 80, enable_tactics: true }),
    });
    const data = await res.json();
    const gtp = data.selected_move;

    if (data.winrate !== undefined) {
      const bWin = Math.round((toMove === "B" ? data.winrate : 1 - data.winrate) * 100);
      document.getElementById("blackWinrate").textContent = `${bWin}%`;
      document.getElementById("whiteWinrate").textContent = `${100 - bWin}%`;
      document.getElementById("winrateBlackFill").style.width = `${bWin}%`;
    }

    const alertEl = document.getElementById("tacticsAlert");
    if (data.tactical_override) {
      alertEl.style.display = "inline-block";
      alertEl.textContent = `Tactics: ${data.tactical_override.replace(/_/g, " ")}`;
    } else {
      alertEl.style.display = "none";
    }

    topCandidates = data.top_candidates || [];
    renderCandidates(topCandidates);

    // Play move
    await fetch(`${API_BASE}/api/board/play`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ color: toMove, coord: gtp }),
    });

    if (data.row !== null && data.col !== null) {
      lastMove = { r: data.row, c: data.col, color: toMove };
    }
    await fetchState();
  } catch (err) {
    console.error("Bot move error:", err);
  }
}

async function analyzePosition() {
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_simulations: 80 }),
    });
    const data = await res.json();
    topCandidates = data.candidates || [];
    renderCandidates(topCandidates);
    drawBoard();
  } catch (err) {
    console.error("Analyze error:", err);
  }
}

function renderCandidates(candidates) {
  const container = document.getElementById("candidatesList");
  container.innerHTML = "";
  candidates.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "candidate-card";
    card.innerHTML = `
      <div>
        <span class="coord">#${i + 1} ${c.gtp_coord}</span>
        <span style="font-size:0.8rem; color:#94a3b8; margin-left:8px;">Visits: ${c.visits}</span>
      </div>
      <div>
        <strong style="color:#38bdf8;">${Math.round(c.winrate * 100)}% Win</strong>
      </div>
    `;
    container.appendChild(card);
  });
}

// Canvas Click Event
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const padding = 25;
  const cell = (canvas.width - 2 * padding) / (boardSize - 1);

  const c = Math.round((x - padding) / cell);
  const r = Math.round((y - padding) / cell);

  if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
    playCoord(r, c);
  }
});

// Controls
document.getElementById("btnBotMove").addEventListener("click", requestBotMove);
document.getElementById("btnAnalyze").addEventListener("click", analyzePosition);
document.getElementById("btnReset").addEventListener("click", async () => {
  await fetch(`${API_BASE}/api/board/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board_size: 19, komi: 7.5 }),
  });
  lastMove = null;
  topCandidates = [];
  document.getElementById("candidatesList").innerHTML = "";
  await fetchState();
});
document.getElementById("btnPass").addEventListener("click", async () => {
  await fetch(`${API_BASE}/api/board/play`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color: toMove, coord: "pass" }),
  });
  await fetchState();
});
document.getElementById("btnResign").addEventListener("click", async () => {
  await fetch(`${API_BASE}/api/board/play`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color: toMove, coord: "resign" }),
  });
  await fetchState();
});

// Initial load
fetchState();
