/**
 * GoBot Extension - Content Script
 * Injects online Go board detection, interactive candidate heatmaps,
 * live winrate evaluation, and floating HUD overlay onto OGS, Fox, KGS, BadukPop, Sabaki, etc.
 */

(function () {
  'use strict';

  // Prevent multiple injections
  if (window.__gobot_injected) return;
  window.__gobot_injected = true;

  console.log('[GoBot] Content script loaded and active.');

  // --- Configuration & State ---
  const state = {
    detectedBoard: null, // { element, type, size: 19, bounds }
    grid: null,          // 19x19 array [0=Empty, 1=Black, 2=White]
    toMove: 'B',
    komi: 7.5,
    overlayCanvas: null,
    overlayCtx: null,
    hudElement: null,
    analysisResult: null,
    isAnalyzing: false,
    autoAnalyze: true,
    showHeatmap: true,
    gameMode: 'advisor', // 'advisor' | 'bot_assist'
    hoveredCandidate: null,
    backendStatus: 'pure_js_fallback',
    animationFrameId: null
  };

  // Standard GTP Column Names
  const GTP_COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';

  function coordToGTP(r, c, size = 19) {
    if (r < 0 || c < 0 || r >= size || c >= size) return 'pass';
    return `${GTP_COLS[c]}${size - r}`;
  }

  // --- Board Detectors ---
  function detectBoardElement() {
    // 1. Online-Go.com (OGS)
    const ogsCanvas = document.querySelector('canvas.board-canvas') || 
                      document.querySelector('div.goban-container canvas') || 
                      document.querySelector('div[class*="goban"] canvas') ||
                      document.querySelector('svg.goban');
    if (ogsCanvas) {
      return { element: ogsCanvas, type: 'OGS', size: 19 };
    }

    // 2. BadukPop / Fox Online / KGS Web / Sabaki Web
    const namedBoards = document.querySelector('.go-board, #go-board, #goban, .goban-canvas, .board-container canvas, #main-board canvas');
    if (namedBoards) {
      return { element: namedBoards, type: 'GENERIC_NAMED', size: 19 };
    }

    // 3. Generic Square Canvas on Page
    const canvases = Array.from(document.querySelectorAll('canvas'));
    for (const c of canvases) {
      const rect = c.getBoundingClientRect();
      if (rect.width >= 220 && rect.height >= 220) {
        const aspectRatio = rect.width / rect.height;
        if (aspectRatio >= 0.85 && aspectRatio <= 1.15) {
          return { element: c, type: 'CANVAS_AUTODETECT', size: 19 };
        }
      }
    }

    // 4. SVG Go Boards
    const svgs = Array.from(document.querySelectorAll('svg'));
    for (const s of svgs) {
      const rect = s.getBoundingClientRect();
      if (rect.width >= 220 && rect.height >= 220) {
        return { element: s, type: 'SVG_AUTODETECT', size: 19 };
      }
    }

    return null;
  }

  // --- Board State Extractor (DOM & Canvas Pixel Inspection) ---
  function extractBoardGrid(boardInfo) {
    const size = boardInfo.size || 19;
    const grid = Array.from({ length: size }, () => Array(size).fill(0));
    const el = boardInfo.element;

    // A. If Canvas: Pixel sampling
    if (el && el.tagName === 'CANVAS') {
      try {
        const ctx = el.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          const width = el.width || el.clientWidth;
          const height = el.height || el.clientHeight;
          const cellW = width / (size + 1);
          const cellH = height / (size + 1);

          for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
              const x = Math.round((c + 1) * cellW);
              const y = Math.round((r + 1) * cellH);
              const pixel = ctx.getImageData(x, y, 1, 1).data;
              const [red, green, blue, alpha] = pixel;

              if (alpha > 50) {
                const lum = 0.299 * red + 0.587 * green + 0.114 * blue;
                if (lum < 55) {
                  grid[r][c] = 1; // Black stone
                } else if (lum > 195) {
                  grid[r][c] = 2; // White stone
                }
              }
            }
          }
        }
      } catch (e) {
        console.warn('[GoBot] Pixel sampling canvas restricted or failed (CORS), checking DOM heuristics');
      }
    }

    // B. DOM stone inspection (e.g. OGS/WGo stone elements with class/coordinates)
    const stoneElements = document.querySelectorAll('.stone, [class*="stone-"], [data-coord]');
    if (stoneElements.length > 0) {
      stoneElements.forEach(stone => {
        const isBlack = stone.classList.contains('black') || stone.className.includes('stone-black') || stone.getAttribute('data-color') === 'black';
        const isWhite = stone.classList.contains('white') || stone.className.includes('stone-white') || stone.getAttribute('data-color') === 'white';
        const coord = stone.getAttribute('data-coord') || stone.id;
        if (coord && (isBlack || isWhite)) {
          // Parse coordinate
          const match = coord.match(/([A-T])([0-9]+)/i);
          if (match) {
            const col = GTP_COLS.indexOf(match[1].toUpperCase());
            const row = size - parseInt(match[2], 10);
            if (col >= 0 && col < size && row >= 0 && row < size) {
              grid[row][col] = isBlack ? 1 : 2;
            }
          }
        }
      });
    }

    // Determine whose turn it is based on stone count if not explicit
    let bCount = 0;
    let wCount = 0;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (grid[r][c] === 1) bCount++;
        else if (grid[r][c] === 2) wCount++;
      }
    }
    state.toMove = bCount <= wCount ? 'B' : 'W';
    return grid;
  }

  // --- Overlay Canvas Manager ---
  function setupOverlayCanvas(boardInfo) {
    if (state.overlayCanvas) {
      state.overlayCanvas.remove();
      state.overlayCanvas = null;
    }

    const targetEl = boardInfo.element;
    const rect = targetEl.getBoundingClientRect();

    const canvas = document.createElement('canvas');
    canvas.id = 'gobot-board-overlay';
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);
    canvas.style.position = 'absolute';
    canvas.style.left = `${window.scrollX + rect.left}px`;
    canvas.style.top = `${window.scrollY + rect.top}px`;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    canvas.style.zIndex = '999990';
    canvas.style.pointerEvents = 'none';

    document.body.appendChild(canvas);
    state.overlayCanvas = canvas;
    state.overlayCtx = canvas.getContext('2d');

    // Update positions on resize / scroll
    window.addEventListener('resize', updateOverlayPosition);
    window.addEventListener('scroll', updateOverlayPosition);
  }

  function updateOverlayPosition() {
    if (!state.overlayCanvas || !state.detectedBoard || !state.detectedBoard.element) return;
    const rect = state.detectedBoard.element.getBoundingClientRect();
    state.overlayCanvas.style.left = `${window.scrollX + rect.left}px`;
    state.overlayCanvas.style.top = `${window.scrollY + rect.top}px`;
    state.overlayCanvas.style.width = `${rect.width}px`;
    state.overlayCanvas.style.height = `${rect.height}px`;
  }

  // --- Overlay Renderer (Candidate Heatmaps, Optimal Badges, Halo) ---
  let animPulse = 0;
  function renderOverlay() {
    if (!state.overlayCtx || !state.overlayCanvas || !state.showHeatmap) {
      if (state.overlayCtx) {
        state.overlayCtx.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
      }
      return;
    }

    const ctx = state.overlayCtx;
    const width = state.overlayCanvas.width;
    const height = state.overlayCanvas.height;
    const size = state.detectedBoard ? state.detectedBoard.size : 19;
    const cellW = width / (size + 1);
    const cellH = height / (size + 1);

    ctx.clearRect(0, 0, width, height);
    animPulse = (animPulse + 0.04) % (Math.PI * 2);

    if (!state.analysisResult || !state.analysisResult.top_candidates) return;

    const candidates = state.analysisResult.top_candidates;
    const maxVisits = Math.max(...candidates.map(c => c.visits || 1), 1);

    // Draw Heatmap / Candidate Points
    candidates.forEach((cand, idx) => {
      if (cand.row === null || cand.col === null || cand.row < 0 || cand.col < 0) return;

      const cx = (cand.col + 1) * cellW;
      const cy = (cand.row + 1) * cellH;
      const radius = Math.min(cellW, cellH) * 0.42;

      // Color coding based on rank and winrate
      const isTop = idx === 0;
      let fillColor = isTop ? 'rgba(16, 185, 129, 0.75)' : // Emerald #1
                      idx === 1 ? 'rgba(59, 130, 246, 0.70)' : // Blue #2
                      idx === 2 ? 'rgba(168, 85, 247, 0.65)' : // Purple #3
                      'rgba(245, 158, 11, 0.55)'; // Amber other

      // Animated Glowing Halo for #1 Optimal Move
      if (isTop) {
        const pulseScale = 1 + Math.sin(animPulse) * 0.15;
        const gradient = ctx.createRadialGradient(cx, cy, radius * 0.5, cx, cy, radius * 1.6 * pulseScale);
        gradient.addColorStop(0, 'rgba(16, 185, 129, 0.8)');
        gradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.35)');
        gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

        ctx.beginPath();
        ctx.arc(cx, cy, radius * 1.6 * pulseScale, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Pulsing border ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 1.15 * pulseScale, 0, Math.PI * 2);
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.5 * (window.devicePixelRatio || 1);
        ctx.stroke();
      }

      // Candidate Marker Circle
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.shadowColor = 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = 6;
      ctx.fill();
      ctx.shadowBlur = 0;

      // White outline
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
      ctx.stroke();

      // Text Badge: Rank `#1` & Winrate %
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const fontSize = Math.max(10, Math.round(radius * 0.75));
      ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;

      const winPct = `${Math.round((cand.winrate || 0.5) * 100)}%`;
      if (radius > 16) {
        ctx.fillText(`#${idx + 1}`, cx, cy - radius * 0.22);
        ctx.font = `600 ${Math.round(fontSize * 0.72)}px -apple-system, BlinkMacSystemFont, sans-serif`;
        ctx.fillText(winPct, cx, cy + radius * 0.35);
      } else {
        ctx.fillText(`${idx + 1}`, cx, cy);
      }
    });
  }

  function startAnimationLoop() {
    if (state.animationFrameId) cancelAnimationFrame(state.animationFrameId);
    function loop() {
      renderOverlay();
      state.animationFrameId = requestAnimationFrame(loop);
    }
    loop();
  }

  // --- Floating HUD Widget ---
  function createHUD() {
    if (state.hudElement) return;

    const hud = document.createElement('div');
    hud.id = 'gobot-hud-widget';
    hud.className = 'gobot-hud-container';
    hud.innerHTML = `
      <div class="gobot-hud-header" id="gobot-hud-drag">
        <div class="gobot-hud-title">
          <span class="gobot-hud-logo-icon">⚫⚪</span>
          <span class="gobot-hud-brand">GoBot Live</span>
          <span class="gobot-hud-badge" id="gobot-hud-badge">PURE JS</span>
        </div>
        <div class="gobot-hud-actions">
          <button class="gobot-hud-btn-icon" id="gobot-btn-minimize" title="Minimize">━</button>
        </div>
      </div>

      <div class="gobot-hud-body" id="gobot-hud-body">
        <!-- Winrate Evaluation Split Bar -->
        <div class="gobot-hud-winrate-section">
          <div class="gobot-hud-winrate-labels">
            <span class="gobot-label-b" id="gobot-wr-b">B 50.0%</span>
            <span class="gobot-label-score" id="gobot-score-est">Even</span>
            <span class="gobot-label-w" id="gobot-wr-w">W 50.0%</span>
          </div>
          <div class="gobot-hud-bar-track">
            <div class="gobot-hud-bar-b" id="gobot-bar-fill-b" style="width: 50%;"></div>
            <div class="gobot-hud-bar-w" id="gobot-bar-fill-w" style="width: 50%;"></div>
          </div>
        </div>

        <!-- Top Candidates Row -->
        <div class="gobot-hud-candidates-row" id="gobot-candidates-container">
          <div class="gobot-cand-pill gobot-top-cand" id="gobot-cand-1">
            <span class="cand-rank">#1</span>
            <span class="cand-coord" id="gobot-c1-move">--</span>
            <span class="cand-pct" id="gobot-c1-pct">--%</span>
          </div>
          <div class="gobot-cand-pill" id="gobot-cand-2">
            <span class="cand-rank">#2</span>
            <span class="cand-coord" id="gobot-c2-move">--</span>
            <span class="cand-pct" id="gobot-c2-pct">--%</span>
          </div>
          <div class="gobot-cand-pill" id="gobot-cand-3">
            <span class="cand-rank">#3</span>
            <span class="cand-coord" id="gobot-c3-move">--</span>
            <span class="cand-pct" id="gobot-c3-pct">--%</span>
          </div>
        </div>

        <!-- Tactical Reason / Desc -->
        <div class="gobot-hud-desc" id="gobot-tactical-desc">
          Ready to analyze position.
        </div>

        <!-- Control Toolbar -->
        <div class="gobot-hud-toolbar">
          <button class="gobot-hud-action-btn primary" id="gobot-btn-analyze">
            <span class="icon">⚡</span> Analyze
          </button>
          <button class="gobot-hud-action-btn" id="gobot-btn-toggle-heatmap">
            <span class="icon">👁️</span> Heatmap: ON
          </button>
          <button class="gobot-hud-action-btn" id="gobot-btn-toggle-mode">
            <span class="icon">🛡️</span> Advisor
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(hud);
    state.hudElement = hud;

    // Attach Event Listeners
    document.getElementById('gobot-btn-analyze').addEventListener('click', () => runAnalysis());
    document.getElementById('gobot-btn-toggle-heatmap').addEventListener('click', toggleHeatmap);
    document.getElementById('gobot-btn-toggle-mode').addEventListener('click', toggleGameMode);
    document.getElementById('gobot-btn-minimize').addEventListener('click', toggleMinimize);

    // Interactive candidate click in Bot Assist mode
    document.getElementById('gobot-candidates-container').addEventListener('click', (e) => {
      const pill = e.target.closest('.gobot-cand-pill');
      if (pill && state.analysisResult && state.analysisResult.top_candidates) {
        const id = pill.id;
        const idx = id === 'gobot-cand-1' ? 0 : id === 'gobot-cand-2' ? 1 : 2;
        const cand = state.analysisResult.top_candidates[idx];
        if (cand && state.gameMode === 'bot_assist') {
          playCandidateMove(cand);
        }
      }
    });

    // Make HUD draggable
    setupDraggableHUD(hud);
  }

  function setupDraggableHUD(el) {
    const dragHandle = document.getElementById('gobot-hud-drag');
    let isDragging = false;
    let startX = 0, startY = 0;
    let initialX = 0, initialY = 0;

    dragHandle.addEventListener('mousedown', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = el.getBoundingClientRect();
      initialX = rect.left;
      initialY = rect.top;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    });

    function onMouseMove(e) {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      el.style.left = `${Math.max(10, initialX + dx)}px`;
      el.style.top = `${Math.max(10, initialY + dy)}px`;
      el.style.right = 'auto';
      el.style.bottom = 'auto';
    }

    function onMouseUp() {
      isDragging = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }
  }

  function toggleHeatmap() {
    state.showHeatmap = !state.showHeatmap;
    const btn = document.getElementById('gobot-btn-toggle-heatmap');
    if (btn) {
      btn.innerHTML = state.showHeatmap ? '<span class="icon">👁️</span> Heatmap: ON' : '<span class="icon">🕶️</span> Heatmap: OFF';
    }
  }

  function toggleGameMode() {
    state.gameMode = state.gameMode === 'advisor' ? 'bot_assist' : 'advisor';
    const btn = document.getElementById('gobot-btn-toggle-mode');
    if (btn) {
      btn.innerHTML = state.gameMode === 'advisor' ? '<span class="icon">🛡️</span> Advisor' : '<span class="icon">🤖</span> Bot Assist';
      btn.classList.toggle('assist-active', state.gameMode === 'bot_assist');
    }
  }

  function toggleMinimize() {
    const body = document.getElementById('gobot-hud-body');
    const btn = document.getElementById('gobot-btn-minimize');
    if (body.style.display === 'none') {
      body.style.display = 'flex';
      btn.innerText = '━';
    } else {
      body.style.display = 'none';
      btn.innerText = '□';
    }
  }

  // --- Assisted Click / Move Trigger ---
  function playCandidateMove(candidate) {
    if (!candidate || candidate.row === null || candidate.col === null || !state.detectedBoard) return;
    const el = state.detectedBoard.element;
    const rect = el.getBoundingClientRect();
    const size = state.detectedBoard.size || 19;
    const cellW = rect.width / (size + 1);
    const cellH = rect.height / (size + 1);

    const targetX = rect.left + (candidate.col + 1) * cellW;
    const targetY = rect.top + (candidate.row + 1) * cellH;

    const eventOptions = {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: targetX,
      clientY: targetY
    };

    el.dispatchEvent(new MouseEvent('mousedown', eventOptions));
    el.dispatchEvent(new MouseEvent('mouseup', eventOptions));
    el.dispatchEvent(new MouseEvent('click', eventOptions));
    console.log(`[GoBot] Assisted move injected at ${candidate.move} (${candidate.row}, ${candidate.col})`);
  }

  // --- Run Position Analysis via Background Engine ---
  async function runAnalysis() {
    if (state.isAnalyzing) return;
    const boardInfo = detectBoardElement();
    if (!boardInfo) {
      updateHUDDescription('No active Go board detected on this page.');
      return;
    }

    state.detectedBoard = boardInfo;
    setupOverlayCanvas(boardInfo);

    state.grid = extractBoardGrid(boardInfo);
    state.isAnalyzing = true;
    updateHUDDescription('Analyzing board candidate moves...');

    try {
      chrome.runtime.sendMessage({
        action: 'OPTIMIZE_MOVE',
        params: {
          board_size: boardInfo.size,
          komi: state.komi,
          grid: state.grid,
          to_move: state.toMove,
          num_simulations: 80,
          enable_tactics: true
        }
      }, (resp) => {
        state.isAnalyzing = false;
        if (resp && resp.ok && resp.result) {
          state.analysisResult = resp.result;
          state.backendStatus = resp.result.engine_backend || 'pure_js_fallback';
          updateHUDWithResult(resp.result);
        } else {
          updateHUDDescription('Engine analysis completed or timed out.');
        }
      });
    } catch (err) {
      state.isAnalyzing = false;
      updateHUDDescription('Error communicating with GoBot engine.');
    }
  }

  // --- Update HUD UI with Engine Results ---
  function updateHUDWithResult(result) {
    // 1. Update Badge
    const badge = document.getElementById('gobot-hud-badge');
    if (badge) {
      badge.innerText = result.engine_backend === 'fastapi' ? 'FASTAPI' : 'PURE JS';
      badge.className = `gobot-hud-badge ${result.engine_backend === 'fastapi' ? 'online' : 'fallback'}`;
    }

    // 2. Update Winrate & Score
    const winrate = result.winrate !== undefined ? result.winrate : 0.5;
    const bWr = state.toMove === 'B' ? winrate : (1 - winrate);
    const wWr = 1 - bWr;

    const bPct = Math.round(bWr * 100);
    const wPct = 100 - bPct;

    const elWrB = document.getElementById('gobot-wr-b');
    const elWrW = document.getElementById('gobot-wr-w');
    const elBarB = document.getElementById('gobot-bar-fill-b');
    const elBarW = document.getElementById('gobot-bar-fill-w');

    if (elWrB) elWrB.innerText = `B ${bPct}%`;
    if (elWrW) elWrW.innerText = `W ${wPct}%`;
    if (elBarB) elBarB.style.width = `${bPct}%`;
    if (elBarW) elBarW.style.width = `${wPct}%`;

    // 3. Update Candidates
    const candidates = result.top_candidates || [];
    for (let i = 1; i <= 3; i++) {
      const cand = candidates[i - 1];
      const moveEl = document.getElementById(`gobot-c${i}-move`);
      const pctEl = document.getElementById(`gobot-c${i}-pct`);
      if (moveEl && pctEl) {
        if (cand) {
          moveEl.innerText = cand.move;
          pctEl.innerText = `${Math.round((cand.winrate || 0.5) * 100)}%`;
        } else {
          moveEl.innerText = '--';
          pctEl.innerText = '--%';
        }
      }
    }

    // 4. Update Description / Tactical Reason
    const best = candidates[0];
    const descText = best ? `${best.move}: ${best.description || 'Optimal candidate'}` : 'Position evaluated.';
    updateHUDDescription(descText);
  }

  function updateHUDDescription(text) {
    const descEl = document.getElementById('gobot-tactical-desc');
    if (descEl) descEl.innerText = text;
  }

  // --- Auto-Observer for Dynamic Board Changes ---
  function setupBoardObserver() {
    const observer = new MutationObserver(() => {
      if (state.autoAnalyze && !state.isAnalyzing) {
        clearTimeout(state._debounceTimer);
        state._debounceTimer = setTimeout(() => {
          const board = detectBoardElement();
          if (board) {
            runAnalysis();
          }
        }, 600);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class', 'data-coord']
    });
  }

  // --- Initialize Content Script ---
  function init() {
    const board = detectBoardElement();
    if (board) {
      state.detectedBoard = board;
      setupOverlayCanvas(board);
      createHUD();
      startAnimationLoop();
      setupBoardObserver();
      setTimeout(runAnalysis, 800);
    } else {
      // Check again after page finishes rendering
      setTimeout(() => {
        const retryBoard = detectBoardElement();
        if (retryBoard) {
          state.detectedBoard = retryBoard;
          setupOverlayCanvas(retryBoard);
          createHUD();
          startAnimationLoop();
          setupBoardObserver();
          setTimeout(runAnalysis, 500);
        }
      }, 1500);
    }
  }

  // --- Message Dispatcher from Background / Popup ---
  chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
    if (req.action === 'TRIGGER_MANUAL_ANALYZE') {
      const board = detectBoardElement();
      if (board) {
        if (!state.hudElement) createHUD();
        runAnalysis();
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: 'No board found' });
      }
      return true;
    }

    if (req.action === 'TOGGLE_OVERLAY_HUD') {
      if (state.hudElement) {
        state.hudElement.classList.toggle('gobot-hidden');
      } else {
        createHUD();
      }
      sendResponse({ ok: true });
      return true;
    }

    if (req.action === 'EXTRACT_BOARD_STATE') {
      const board = detectBoardElement();
      if (board) {
        const grid = extractBoardGrid(board);
        sendResponse({
          ok: true,
          board_size: board.size,
          grid,
          to_move: state.toMove
        });
      } else {
        sendResponse({ ok: false, error: 'No board detected' });
      }
      return true;
    }

    return false;
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
