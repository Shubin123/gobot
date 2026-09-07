/**
 * GoBot Extension - Popup Controller
 * Manages Tab switching, Live Advisor sync, Interactive Studio Board, and Extension Settings.
 */

document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // --- State ---
  const state = {
    activeTab: 'tab-advisor',
    backendStatus: 'checking', // 'fastapi' | 'pure_js' | 'error'
    studioSize: 19,
    studioTool: 1, // 1=Black, 2=White, 0=Empty
    studioGrid: Array.from({ length: 19 }, () => Array(19).fill(0)),
    studioToMove: 'B',
    studioCandidates: [],
    settings: {
      serverUrl: 'http://localhost:8000',
      numSimulations: 60,
      enableTactics: true,
      forcePureJS: false,
      autoAnalyze: true
    }
  };

  // --- Element References ---
  const tabs = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const engineBadge = document.getElementById('engine-status-badge');
  const engineStatusText = document.getElementById('engine-status-text');

  // Advisor Elements
  const advisorWrB = document.getElementById('advisor-wr-b');
  const advisorWrW = document.getElementById('advisor-wr-w');
  const advisorBarB = document.getElementById('advisor-bar-b');
  const advisorBarW = document.getElementById('advisor-bar-w');
  const advisorScoreEst = document.getElementById('advisor-score-est');
  const spotlightCoord = document.getElementById('spotlight-coord');
  const spotlightWinrate = document.getElementById('spotlight-winrate');
  const spotlightDesc = document.getElementById('spotlight-desc');
  const candidatesList = document.getElementById('candidates-list');
  const toMoveLabel = document.getElementById('to-move-label');
  const btnSyncAnalyze = document.getElementById('btn-sync-analyze');
  const btnToggleHUD = document.getElementById('btn-toggle-hud');

  // Studio Elements
  const studioCanvas = document.getElementById('studio-board-canvas');
  const studioCtx = studioCanvas ? studioCanvas.getContext('2d') : null;
  const studioSizeSelect = document.getElementById('studio-size-select');
  const toolBlack = document.getElementById('tool-black');
  const toolWhite = document.getElementById('tool-white');
  const toolErase = document.getElementById('tool-erase');
  const btnStudioAnalyze = document.getElementById('btn-studio-analyze');
  const btnStudioClear = document.getElementById('btn-studio-clear');
  const btnStudioStarpoints = document.getElementById('btn-studio-starpoints');
  const studioTurnInd = document.getElementById('studio-turn-indicator');
  const studioScoreInd = document.getElementById('studio-score-indicator');

  // Settings Elements
  const settingsForm = document.getElementById('settings-form');
  const settingServerUrl = document.getElementById('setting-server-url');
  const settingSimulations = document.getElementById('setting-simulations');
  const simValDisplay = document.getElementById('sim-val-display');
  const settingEnableTactics = document.getElementById('setting-enable-tactics');
  const settingForcePureJS = document.getElementById('setting-force-pure-js');
  const settingAutoAnalyze = document.getElementById('setting-auto-analyze');
  const btnReconnectEngine = document.getElementById('btn-reconnect-engine');

  // --- Initialize ---
  initTabs();
  initSettings();
  checkEngineStatus();
  initStudioBoard();
  syncFromActiveTab();

  // --- Tab Navigation ---
  function initTabs() {
    tabs.forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));

        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const targetContent = document.getElementById(targetId);
        if (targetContent) {
          targetContent.classList.add('active');
          state.activeTab = targetId;
          if (targetId === 'tab-manual') {
            drawStudioBoard();
          }
        }
      });
    });
  }

  // --- Engine Status Check ---
  function checkEngineStatus() {
    engineStatusText.innerText = 'Checking...';
    engineBadge.className = 'engine-badge';

    chrome.runtime.sendMessage({ action: 'PING_ENGINE' }, (resp) => {
      if (resp && resp.ok && resp.health) {
        if (resp.health.mode === 'fastapi') {
          engineBadge.className = 'engine-badge fastapi';
          engineStatusText.innerText = 'FastAPI Connected';
        } else {
          engineBadge.className = 'engine-badge pure-js';
          engineStatusText.innerText = 'Pure JS Fallback';
        }
      } else {
        engineBadge.className = 'engine-badge pure-js';
        engineStatusText.innerText = 'Pure JS Fallback';
      }
    });
  }

  // --- Live Advisor Logic ---
  btnSyncAnalyze.addEventListener('click', () => syncFromActiveTab());
  btnToggleHUD.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'TOGGLE_OVERLAY_HUD' });
      }
    });
  });

  function syncFromActiveTab() {
    spotlightDesc.innerText = 'Analyzing active board...';
    btnSyncAnalyze.disabled = true;

    chrome.tabs.query({ active: true, currentWindow: true }, (tabsList) => {
      if (!tabsList || !tabsList[0] || !tabsList[0].id) {
        btnSyncAnalyze.disabled = false;
        spotlightDesc.innerText = 'No active Go game detected in browser tab.';
        return;
      }

      chrome.tabs.sendMessage(tabsList[0].id, { action: 'EXTRACT_BOARD_STATE' }, (resp) => {
        if (resp && resp.ok && resp.grid) {
          toMoveLabel.innerText = `Turn: ${resp.to_move === 'B' ? 'Black' : 'White'}`;
          // Request optimization
          chrome.runtime.sendMessage({
            action: 'OPTIMIZE_MOVE',
            params: {
              board_size: resp.board_size || 19,
              grid: resp.grid,
              to_move: resp.to_move || 'B',
              num_simulations: state.settings.numSimulations,
              enable_tactics: state.settings.enableTactics
            }
          }, (optResp) => {
            btnSyncAnalyze.disabled = false;
            if (optResp && optResp.ok && optResp.result) {
              renderAdvisorResults(optResp.result, resp.to_move || 'B');
            } else {
              spotlightDesc.innerText = 'Failed to analyze active board.';
            }
          });
        } else {
          // No in-page board detected, evaluate standard initial 19x19 star openings
          btnSyncAnalyze.disabled = false;
          chrome.runtime.sendMessage({
            action: 'OPTIMIZE_MOVE',
            params: {
              board_size: 19,
              to_move: 'B',
              num_simulations: state.settings.numSimulations,
              enable_tactics: state.settings.enableTactics
            }
          }, (optResp) => {
            if (optResp && optResp.ok && optResp.result) {
              renderAdvisorResults(optResp.result, 'B');
              spotlightDesc.innerText = 'Standard 19x19 Opening Analysis (Tab offline)';
            }
          });
        }
      });
    });
  }

  function renderAdvisorResults(result, toMove = 'B') {
    const winrate = result.winrate !== undefined ? result.winrate : 0.5;
    const bWr = toMove === 'B' ? winrate : (1 - winrate);
    const wWr = 1 - bWr;
    const bPct = Math.round(bWr * 100);
    const wPct = 100 - bPct;

    advisorWrB.innerText = `Black ${bPct}%`;
    advisorWrW.innerText = `White ${wPct}%`;
    advisorBarB.style.width = `${bPct}%`;
    advisorBarW.style.width = `${wPct}%`;

    const diff = Math.round((bWr - 0.5) * 40);
    advisorScoreEst.innerText = diff > 0 ? `B+${diff}` : diff < 0 ? `W+${Math.abs(diff)}` : 'Even';

    spotlightCoord.innerText = result.selected_move || 'pass';
    spotlightWinrate.innerText = `${Math.round(winrate * 100)}% Winrate`;

    const candidates = result.top_candidates || [];
    const topCand = candidates[0];
    spotlightDesc.innerText = topCand ? (topCand.description || 'Optimal candidate move') : 'Best position evaluated.';

    // Candidates List
    candidatesList.innerHTML = '';
    if (candidates.length === 0) {
      candidatesList.innerHTML = '<div class="empty-state">No candidates found.</div>';
      return;
    }

    candidates.forEach((cand, idx) => {
      const item = document.createElement('div');
      item.className = 'candidate-item';
      item.innerHTML = `
        <div class="cand-left">
          <span class="cand-rank-tag">#${idx + 1}</span>
          <span class="cand-move-name">${cand.move}</span>
          <span class="cand-desc-text">${cand.description || ''}</span>
        </div>
        <div class="cand-right">
          <span class="cand-winrate-tag">${Math.round((cand.winrate || 0.5) * 100)}%</span>
          <span class="cand-visits-tag">${cand.visits || 0} sims</span>
        </div>
      `;
      candidatesList.appendChild(item);
    });
  }

  // --- Interactive Board Studio ---
  function initStudioBoard() {
    if (!studioCanvas || !studioCtx) return;

    resizeStudioGrid(19);

    studioSizeSelect.addEventListener('change', (e) => {
      const size = parseInt(e.target.value, 10);
      resizeStudioGrid(size);
      drawStudioBoard();
    });

    [toolBlack, toolWhite, toolErase].forEach(btn => {
      btn.addEventListener('click', () => {
        [toolBlack, toolWhite, toolErase].forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.studioTool = parseInt(btn.getAttribute('data-color'), 10);
      });
    });

    studioCanvas.addEventListener('click', onStudioCanvasClick);

    btnStudioClear.addEventListener('click', () => {
      resizeStudioGrid(state.studioSize);
      state.studioCandidates = [];
      drawStudioBoard();
      updateStudioScore();
    });

    btnStudioStarpoints.addEventListener('click', () => {
      placeStarPoints(state.studioSize);
      drawStudioBoard();
      updateStudioScore();
    });

    btnStudioAnalyze.addEventListener('click', runStudioAnalysis);

    drawStudioBoard();
  }

  function resizeStudioGrid(size) {
    state.studioSize = size;
    state.studioGrid = Array.from({ length: size }, () => Array(size).fill(0));
    state.studioCandidates = [];
  }

  function placeStarPoints(size) {
    resizeStudioGrid(size);
    if (size === 19) {
      const stars = [[3, 3], [3, 15], [15, 3], [15, 15], [9, 9]];
      stars.forEach(([r, c], idx) => {
        state.studioGrid[r][c] = (idx % 2 === 0) ? 1 : 2;
      });
    } else if (size === 9) {
      state.studioGrid[4][4] = 1;
      state.studioGrid[2][2] = 2;
    }
  }

  function onStudioCanvasClick(e) {
    const rect = studioCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const size = state.studioSize;
    const padding = 16;
    const boardWidth = studioCanvas.width - padding * 2;
    const cellSize = boardWidth / (size - 1);

    const c = Math.round((x - padding) / cellSize);
    const r = Math.round((y - padding) / cellSize);

    if (r >= 0 && r < size && c >= 0 && c < size) {
      state.studioGrid[r][c] = state.studioTool;
      state.studioCandidates = [];
      drawStudioBoard();
      updateStudioScore();
    }
  }

  function updateStudioScore() {
    let b = 0, w = 0;
    for (let r = 0; r < state.studioSize; r++) {
      for (let c = 0; c < state.studioSize; c++) {
        if (state.studioGrid[r][c] === 1) b++;
        else if (state.studioGrid[r][c] === 2) w++;
      }
    }
    state.studioToMove = b <= w ? 'B' : 'W';
    studioTurnInd.innerText = `To Move: ${state.studioToMove === 'B' ? 'Black' : 'White'}`;
    const diff = b - w;
    studioScoreInd.innerText = diff >= 0 ? `B+${diff}` : `W+${Math.abs(diff)}`;
  }

  function drawStudioBoard() {
    if (!studioCtx || !studioCanvas) return;
    const ctx = studioCtx;
    const size = state.studioSize;
    const width = studioCanvas.width;
    const height = studioCanvas.height;
    const padding = 16;
    const boardWidth = width - padding * 2;
    const cellSize = boardWidth / (size - 1);

    // Wood Background
    ctx.fillStyle = '#DCB35C';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = '#3E2723';
    ctx.lineWidth = 1;
    for (let i = 0; i < size; i++) {
      // Horizontal
      ctx.beginPath();
      ctx.moveTo(padding, padding + i * cellSize);
      ctx.lineTo(width - padding, padding + i * cellSize);
      ctx.stroke();

      // Vertical
      ctx.beginPath();
      ctx.moveTo(padding + i * cellSize, padding);
      ctx.lineTo(padding + i * cellSize, height - padding);
      ctx.stroke();
    }

    // Star points
    const starCoords = size === 19 ? [
      [3, 3], [3, 9], [3, 15],
      [9, 3], [9, 9], [9, 15],
      [15, 3], [15, 9], [15, 15]
    ] : size === 13 ? [
      [3, 3], [3, 9], [6, 6], [9, 3], [9, 9]
    ] : [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];

    ctx.fillStyle = '#3E2723';
    starCoords.forEach(([r, c]) => {
      ctx.beginPath();
      ctx.arc(padding + c * cellSize, padding + r * cellSize, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Stones
    const stoneRadius = cellSize * 0.46;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const val = state.studioGrid[r][c];
        if (val === 1) {
          // Black Stone
          const sx = padding + c * cellSize;
          const sy = padding + r * cellSize;
          const grad = ctx.createRadialGradient(sx - stoneRadius * 0.3, sy - stoneRadius * 0.3, stoneRadius * 0.1, sx, sy, stoneRadius);
          grad.addColorStop(0, '#555555');
          grad.addColorStop(1, '#111111');
          ctx.beginPath();
          ctx.arc(sx, sy, stoneRadius, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.shadowColor = 'rgba(0,0,0,0.4)';
          ctx.shadowBlur = 4;
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (val === 2) {
          // White Stone
          const sx = padding + c * cellSize;
          const sy = padding + r * cellSize;
          const grad = ctx.createRadialGradient(sx - stoneRadius * 0.3, sy - stoneRadius * 0.3, stoneRadius * 0.1, sx, sy, stoneRadius);
          grad.addColorStop(0, '#FFFFFF');
          grad.addColorStop(1, '#DDDDDD');
          ctx.beginPath();
          ctx.arc(sx, sy, stoneRadius, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.shadowColor = 'rgba(0,0,0,0.3)';
          ctx.shadowBlur = 4;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }
    }

    // Candidate Move Markers on Canvas
    if (state.studioCandidates && state.studioCandidates.length > 0) {
      state.studioCandidates.forEach((cand, idx) => {
        if (cand.row !== null && cand.col !== null && cand.row >= 0 && cand.col >= 0) {
          const cx = padding + cand.col * cellSize;
          const cy = padding + cand.row * cellSize;
          const isTop = idx === 0;

          ctx.beginPath();
          ctx.arc(cx, cy, stoneRadius * 0.85, 0, Math.PI * 2);
          ctx.fillStyle = isTop ? 'rgba(16, 185, 129, 0.85)' : 'rgba(59, 130, 246, 0.75)';
          ctx.fill();
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 9px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`#${idx + 1}`, cx, cy);
        }
      });
    }
  }

  function runStudioAnalysis() {
    btnStudioAnalyze.disabled = true;
    btnStudioAnalyze.innerText = 'Evaluating...';

    chrome.runtime.sendMessage({
      action: 'OPTIMIZE_MOVE',
      params: {
        board_size: state.studioSize,
        grid: state.studioGrid,
        to_move: state.studioToMove,
        num_simulations: state.settings.numSimulations,
        enable_tactics: state.settings.enableTactics
      }
    }, (resp) => {
      btnStudioAnalyze.disabled = false;
      btnStudioAnalyze.innerHTML = '<span class="icon">🔍</span> Evaluate Position';

      if (resp && resp.ok && resp.result) {
        state.studioCandidates = resp.result.top_candidates || [];
        drawStudioBoard();
      }
    });
  }

  // --- Settings Form Handling ---
  function initSettings() {
    settingSimulations.addEventListener('input', (e) => {
      simValDisplay.innerText = e.target.value;
    });

    chrome.runtime.sendMessage({ action: 'GET_SETTINGS' }, (resp) => {
      if (resp && resp.ok && resp.settings) {
        state.settings = resp.settings;
        settingServerUrl.value = resp.settings.serverUrl || 'http://localhost:8000';
        settingSimulations.value = resp.settings.numSimulations || 60;
        simValDisplay.innerText = resp.settings.numSimulations || 60;
        settingEnableTactics.checked = resp.settings.enableTactics !== false;
        settingForcePureJS.checked = !!resp.settings.forcePureJS;
        settingAutoAnalyze.checked = resp.settings.autoAnalyze !== false;
      }
    });

    settingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const updated = {
        serverUrl: settingServerUrl.value.trim(),
        numSimulations: parseInt(settingSimulations.value, 10),
        enableTactics: settingEnableTactics.checked,
        forcePureJS: settingForcePureJS.checked,
        autoAnalyze: settingAutoAnalyze.checked
      };

      chrome.runtime.sendMessage({
        action: 'SAVE_SETTINGS',
        settings: updated
      }, (resp) => {
        if (resp && resp.ok) {
          state.settings = updated;
          checkEngineStatus();
          alert('GoBot settings saved successfully!');
        }
      });
    });

    btnReconnectEngine.addEventListener('click', checkEngineStatus);
  }
});
