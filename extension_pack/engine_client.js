/**
 * GoBot Unified Engine Client
 * Handles communication with local FastAPI backend (http://localhost:8000)
 * with seamless automatic fallback to the embedded Pure-JS MCTS / Joseki Engine
 * and optional ONNX Web Runtime support.
 */

(function (global) {
  'use strict';

  // Import fast pure-JS engine (Node or Browser)
  let FastEngine = global.GoBotFastEngine;
  if (!FastEngine && typeof require !== 'undefined') {
    try {
      FastEngine = require('./fast_mcts.js');
    } catch (e) {
      // Ignored
    }
  }

  class GoBotEngineClient {
    constructor(options = {}) {
      this.serverUrl = options.serverUrl || 'http://localhost:8000';
      this.wsUrl = options.wsUrl || 'ws://localhost:8000/ws/game';
      this.timeoutMs = options.timeoutMs || 2500;
      this.numSimulations = options.numSimulations || 60;
      this.enableTactics = options.enableTactics !== false;
      this.forcePureJS = options.forcePureJS || false;

      this.isServerOnline = false;
      this.lastHealthCheck = 0;
      this.activeBackend = 'unknown'; // 'fastapi' | 'pure_js_fallback' | 'onnx_web'
      this.wsClient = null;

      // Fast JS engine instance
      this.jsEngine = FastEngine ? new FastEngine.FastMCTS({
        numSimulations: this.numSimulations,
        enableTactics: this.enableTactics
      }) : null;
    }

    setServerUrl(url) {
      this.serverUrl = url.replace(/\/+$/, '');
      this.wsUrl = this.serverUrl.replace(/^http/, 'ws') + '/ws/game';
    }

    async checkHealth() {
      if (this.forcePureJS) {
        this.isServerOnline = false;
        this.activeBackend = 'pure_js_fallback';
        return {
          status: 'ok',
          mode: 'pure_js_fallback',
          has_model: false,
          details: 'Pure JS Engine Active (Manual override)'
        };
      }

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1200);

        const resp = await fetch(`${this.serverUrl}/api/health`, {
          method: 'GET',
          signal: controller.signal,
          headers: { 'Accept': 'application/json' }
        });
        clearTimeout(timeoutId);

        if (resp.ok) {
          const data = await resp.json();
          this.isServerOnline = true;
          this.activeBackend = 'fastapi';
          this.lastHealthCheck = Date.now();
          return {
            status: 'ok',
            mode: 'fastapi',
            has_model: data.has_model || false,
            board_size: data.board_size || 19,
            details: 'Local FastAPI Python Engine Connected'
          };
        }
      } catch (err) {
        // Fallback to pure JS
      }

      this.isServerOnline = false;
      this.activeBackend = 'pure_js_fallback';
      this.lastHealthCheck = Date.now();
      return {
        status: 'ok',
        mode: 'pure_js_fallback',
        has_model: false,
        details: 'Pure JS MCTS & Joseki Engine Active (Auto Fallback)'
      };
    }

    async optimizeMove(params = {}) {
      const boardSize = params.board_size || 19;
      const komi = params.komi !== undefined ? params.komi : 7.5;
      const toMove = (params.to_move || 'B').toUpperCase();
      const numSimulations = params.num_simulations || this.numSimulations;
      const enableTactics = params.enable_tactics !== undefined ? params.enable_tactics : this.enableTactics;
      const grid = params.grid || null;

      // Try local FastAPI server if not forced to Pure JS
      if (!this.forcePureJS) {
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

          const payload = {
            board_size: boardSize,
            komi: komi,
            to_move: toMove,
            num_simulations: numSimulations,
            enable_tactics: enableTactics
          };
          if (grid) payload.grid = grid;

          const resp = await fetch(`${this.serverUrl}/api/optimize/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
          });
          clearTimeout(timeoutId);

          if (resp.ok) {
            const data = await resp.json();
            this.isServerOnline = true;
            this.activeBackend = 'fastapi';
            return {
              ...data,
              engine_backend: 'fastapi',
              backend_label: 'Python Neural Engine (FastAPI)'
            };
          }
        } catch (err) {
          // Fallback to pure JS engine below
        }
      }

      // Pure JS MCTS Fallback Execution
      this.isServerOnline = false;
      this.activeBackend = 'pure_js_fallback';

      return this.runPureJSOptimizer({
        boardSize,
        komi,
        toMove,
        grid,
        numSimulations,
        enableTactics
      });
    }

    runPureJSOptimizer({ boardSize, komi, toMove, grid, numSimulations, enableTactics }) {
      if (!FastEngine) {
        throw new Error('GoBot Fast Engine is not available.');
      }

      const board = new FastEngine.Board(boardSize, komi);
      board.to_move = FastEngine.Color.fromString(toMove);

      if (grid && Array.isArray(grid)) {
        for (let r = 0; r < boardSize; r++) {
          for (let c = 0; c < boardSize; c++) {
            const val = grid[r] && grid[r][c] !== undefined ? grid[r][c] : 0;
            board.set(r, c, val);
          }
        }
      }

      const mcts = new FastEngine.FastMCTS({
        numSimulations: numSimulations || this.numSimulations,
        enableTactics: enableTactics
      });

      const result = mcts.search(board, numSimulations || this.numSimulations);

      return {
        ...result,
        engine_backend: 'pure_js_fallback',
        backend_label: 'Pure JavaScript MCTS + Joseki Engine (Embedded)'
      };
    }

    async analyzeBoard(params = {}) {
      const boardSize = params.board_size || 19;
      const komi = params.komi !== undefined ? params.komi : 7.5;
      const toMove = (params.to_move || 'B').toUpperCase();
      const numSimulations = params.num_simulations || this.numSimulations;
      const grid = params.grid || null;

      if (!this.forcePureJS) {
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

          const payload = {
            board_size: boardSize,
            komi: komi,
            to_move: toMove,
            num_simulations: numSimulations
          };
          if (grid) payload.grid = grid;

          const resp = await fetch(`${this.serverUrl}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
          });
          clearTimeout(timeoutId);

          if (resp.ok) {
            const data = await resp.json();
            this.isServerOnline = true;
            this.activeBackend = 'fastapi';
            return {
              ...data,
              engine_backend: 'fastapi'
            };
          }
        } catch (err) {
          // Fallback to pure JS
        }
      }

      // Pure JS Analyze
      if (!FastEngine) {
        throw new Error('GoBot Fast Engine is not available.');
      }

      const board = new FastEngine.Board(boardSize, komi);
      board.to_move = FastEngine.Color.fromString(toMove);

      if (grid && Array.isArray(grid)) {
        for (let r = 0; r < boardSize; r++) {
          for (let c = 0; c < boardSize; c++) {
            const val = grid[r] && grid[r][c] !== undefined ? grid[r][c] : 0;
            board.set(r, c, val);
          }
        }
      }

      const score = board.calculateAreaScore();
      const mcts = new FastEngine.FastMCTS({ numSimulations });
      const optResult = mcts.search(board, numSimulations);

      return {
        current_score_estimate: score.result_str,
        black_territory_count: score.black_score,
        white_territory_count: score.white_score,
        candidates: optResult.top_candidates,
        engine_backend: 'pure_js_fallback'
      };
    }
  }

  // --- Export Module ---
  const GoBotEngineModule = {
    GoBotEngineClient,
    FastEngine
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = GoBotEngineModule;
  }
  if (typeof globalThis !== 'undefined') {
    globalThis.GoBotEngineClient = GoBotEngineClient;
    globalThis.GoBotEngineModule = GoBotEngineModule;
  }
})(typeof self !== 'undefined' ? self : this);
