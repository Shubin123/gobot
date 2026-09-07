/**
 * GoBot GH Pages — Browser Go Engine
 *
 * Architecture:
 *  ModelLoader  — fetches ONNX shards, caches via IndexedDB, progress UI
 *  GoEngine     — ONNX Runtime Web inference (WebGPU → WASM fallback) + pure-JS MCTS
 *  BoardState   — lightweight Go rules (captures, Ko, pass, scoring)
 *  BoardRenderer — Canvas rendering (stones, star points, candidates, last move)
 *  App          — wires everything together, handles UI events
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const GTP_COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const DB_NAME     = 'gobot-model-cache';
const DB_VERSION  = 1;
const STORE_NAME  = 'shards';
const MANIFEST_URL = 'model/manifest.json';
const MODEL_BASE   = 'model/';

// Colour constants matching the Python engine
const EMPTY = 0, BLACK = 1, WHITE = 2;

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

function opponent(c) { return c === BLACK ? WHITE : BLACK; }

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function showToast(msg, type = '', ms = 3000) {
  const el = document.getElementById('statusToast');
  el.textContent = msg;
  el.className = `visible ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.className = '', ms);
}

// ═══════════════════════════════════════════════════════════════════════════
// IndexedDB helpers
// ═══════════════════════════════════════════════════════════════════════════

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => e.target.result.createObjectStore(STORE_NAME);
    req.onsuccess = e => resolve(e.target.result);
    req.onerror = e => reject(e.target.error);
  });
}

async function idbGet(db, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = e => resolve(e.target.result);
    req.onerror = e => reject(e.target.error);
  });
}

async function idbPut(db, key, value) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const req = tx.objectStore(STORE_NAME).put(value, key);
    req.onsuccess = () => resolve();
    req.onerror = e => reject(e.target.error);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ModelLoader — fetch + shard + cache
// ═══════════════════════════════════════════════════════════════════════════

class ModelLoader {
  constructor() {
    this.db = null;
    this.manifest = null;
  }

  _setBanner(visible) {
    document.getElementById('downloadBanner').classList.toggle('visible', visible);
  }

  _setProgress(loaded, total, label) {
    const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
    document.getElementById('progressBar').style.width = `${pct}%`;
    document.getElementById('progressLabel').textContent = label;
  }

  async fetchManifest() {
    const r = await fetch(MANIFEST_URL);
    if (!r.ok) throw new Error(`Failed to fetch manifest: ${r.status}`);
    return r.json();
  }

  /** Fetch a single URL with progress tracking, returns ArrayBuffer */
  async fetchWithProgress(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
    const total = parseInt(r.headers.get('content-length') || '0');
    const reader = r.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total || loaded);
    }
    const combined = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { combined.set(chunk, offset); offset += chunk.length; }
    return combined.buffer;
  }

  /**
   * Load the ONNX model, using IndexedDB cache for shards.
   * Shows download UI on first fetch.
   * Returns ArrayBuffer of reassembled model.
   */
  async load(onStatus) {
    this.db = await openDB();
    this.manifest = await this.fetchManifest();

    const { shards, total_size_bytes: totalBytes } = this.manifest;
    const cacheKey = `model-v${this.manifest.version}`;

    // --- Cache hit ---
    const cached = await idbGet(this.db, cacheKey);
    if (cached) {
      onStatus('Loaded from cache (no download needed)', 'cache');
      return cached;
    }

    // --- Cache miss: show banner + download ---
    this._setBanner(true);
    document.getElementById('bannerSub').textContent =
      `The model (${formatBytes(totalBytes)}) will be downloaded once and cached locally. ` +
      `Future visits load instantly — GPU acceleration enabled automatically.`;

    const shardBuffers = [];
    let downloaded = 0;

    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const shardKey = `shard-v${this.manifest.version}-${i}`;

      // Check individual shard cache
      const cachedShard = await idbGet(this.db, shardKey);
      if (cachedShard) {
        shardBuffers.push(cachedShard);
        downloaded += cachedShard.byteLength;
        this._setProgress(downloaded, totalBytes,
          `Shard ${i + 1}/${shards.length} loaded from cache`);
        continue;
      }

      // Fetch shard
      const url = MODEL_BASE + shard.filename;
      document.getElementById('bannerTitle').textContent =
        `Downloading model shard ${i + 1} of ${shards.length}...`;

      const buf = await this.fetchWithProgress(url, (loaded, total) => {
        this._setProgress(
          downloaded + loaded,
          totalBytes || total * shards.length,
          `Shard ${i + 1}/${shards.length}: ${formatBytes(downloaded + loaded)} / ${formatBytes(totalBytes)}`
        );
      });

      // Cache this shard
      await idbPut(this.db, shardKey, buf);
      shardBuffers.push(buf);
      downloaded += buf.byteLength;
    }

    // Assemble shards
    this._setProgress(totalBytes, totalBytes, 'Assembling model...');
    const assembled = new Uint8Array(totalBytes);
    let offset = 0;
    for (const buf of shardBuffers) {
      assembled.set(new Uint8Array(buf), offset);
      offset += buf.byteLength;
    }
    const modelBuffer = assembled.buffer;

    // Cache the full assembled model
    await idbPut(this.db, cacheKey, modelBuffer);

    this._setProgress(totalBytes, totalBytes, 'Download complete!');
    setTimeout(() => this._setBanner(false), 1500);

    return modelBuffer;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BoardState — Go rules (captures, Ko, legality, scoring)
// ═══════════════════════════════════════════════════════════════════════════

class BoardState {
  constructor(size = 9, komi = 7.5) {
    this.size = size;
    this.komi = komi;
    this.grid = new Int8Array(size * size);  // row-major
    this.toMove = BLACK;
    this.captures = { [BLACK]: 0, [WHITE]: 0 };
    this.koPoint = -1;
    this.history = [];       // for undo
    this.moveCount = 0;
    this.consecutivePasses = 0;
    this.isGameOver = false;
    this.lastMove = -1;
  }

  idx(r, c) { return r * this.size + c; }
  get(r, c) { return this.grid[this.idx(r, c)]; }
  set(r, c, v) { this.grid[this.idx(r, c)] = v; }

  adjacent(r, c) {
    const { size } = this;
    const res = [];
    if (r > 0)        res.push([r - 1, c]);
    if (r < size - 1) res.push([r + 1, c]);
    if (c > 0)        res.push([r, c - 1]);
    if (c < size - 1) res.push([r, c + 1]);
    return res;
  }

  /** BFS to find group + liberties */
  groupInfo(r, c) {
    const color = this.get(r, c);
    if (color === EMPTY) return null;
    const seen = new Set();
    const liberties = new Set();
    const queue = [[r, c]];
    seen.add(this.idx(r, c));
    while (queue.length) {
      const [cr, cc] = queue.pop();
      for (const [nr, nc] of this.adjacent(cr, cc)) {
        const ni = this.idx(nr, nc);
        const nc_color = this.grid[ni];
        if (nc_color === EMPTY) {
          liberties.add(ni);
        } else if (nc_color === color && !seen.has(ni)) {
          seen.add(ni);
          queue.push([nr, nc]);
        }
      }
    }
    return { stones: seen, liberties };
  }

  _captureIfDead(r, c, capturingColor) {
    if (this.get(r, c) !== opponent(capturingColor)) return 0;
    const info = this.groupInfo(r, c);
    if (info.liberties.size === 0) {
      for (const i of info.stones) this.grid[i] = EMPTY;
      this.captures[capturingColor] += info.stones.size;
      return info.stones.size;
    }
    return 0;
  }

  isLegal(r, c, color) {
    if (this.isGameOver) return false;
    const i = this.idx(r, c);
    if (this.grid[i] !== EMPTY) return false;
    if (i === this.koPoint) return false;

    // Test play
    const saved = this.grid.slice();
    const savedCap = { ...this.captures };
    const savedKo = this.koPoint;

    this.grid[i] = color;
    let captured = 0;
    for (const [nr, nc] of this.adjacent(r, c)) {
      captured += this._captureIfDead(nr, nc, color);
    }
    const info = this.groupInfo(r, c);
    const legal = info && info.liberties.size > 0;

    // Restore
    this.grid.set(saved);
    this.captures = savedCap;
    this.koPoint = savedKo;

    return legal;
  }

  play(r, c, color) {
    // Pass
    if (r < 0) {
      this.history.push({ type: 'pass', color, koPoint: this.koPoint, move: -1 });
      this.consecutivePasses++;
      if (this.consecutivePasses >= 2) this.isGameOver = true;
      this.toMove = opponent(color);
      this.lastMove = -1;
      this.moveCount++;
      return true;
    }

    if (!this.isLegal(r, c, color)) return false;

    const i = this.idx(r, c);
    const snapshot = {
      type: 'move',
      color,
      r, c,
      grid: this.grid.slice(),
      captures: { ...this.captures },
      koPoint: this.koPoint,
      consecutivePasses: this.consecutivePasses,
      lastMove: this.lastMove,
    };
    this.history.push(snapshot);

    this.grid[i] = color;
    this.consecutivePasses = 0;

    let captured = 0;
    let newKo = -1;
    for (const [nr, nc] of this.adjacent(r, c)) {
      const info_before = this.groupInfo(nr, nc);
      const cap = this._captureIfDead(nr, nc, color);
      if (cap === 1 && captured === 0) {
        // Potential Ko point = the captured stone's position
        for (const [ar, ac] of this.adjacent(r, c)) {
          if (this.get(ar, ac) === EMPTY) {
            newKo = this.idx(ar, ac);
          }
        }
      }
      captured += cap;
    }

    this.koPoint = (captured === 1) ? newKo : -1;
    this.lastMove = i;
    this.toMove = opponent(color);
    this.moveCount++;
    return true;
  }

  undo() {
    if (!this.history.length) return false;
    const snap = this.history.pop();
    if (snap.type === 'move') {
      this.grid.set(snap.grid);
      this.captures = snap.captures;
      this.koPoint = snap.koPoint;
      this.consecutivePasses = snap.consecutivePasses;
      this.lastMove = snap.lastMove;
    } else {
      this.consecutivePasses = Math.max(0, this.consecutivePasses - 1);
      this.koPoint = snap.koPoint;
    }
    this.toMove = snap.color;
    this.moveCount--;
    this.isGameOver = false;
    return true;
  }

  /** Simple territory count (no flood-fill — area scoring with current stones) */
  scoreEstimate() {
    let black = this.captures[BLACK];
    let white = this.captures[WHITE] + this.komi;
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const v = this.get(r, c);
        if (v === BLACK) black++;
        else if (v === WHITE) white++;
      }
    }
    const diff = black - white;
    if (diff > 0) return `B+${diff.toFixed(1)}`;
    if (diff < 0) return `W+${(-diff).toFixed(1)}`;
    return 'Jigo';
  }

  clone() {
    const b = new BoardState(this.size, this.komi);
    b.grid.set(this.grid);
    b.toMove = this.toMove;
    b.captures = { ...this.captures };
    b.koPoint = this.koPoint;
    b.consecutivePasses = this.consecutivePasses;
    b.isGameOver = this.isGameOver;
    b.lastMove = this.lastMove;
    b.moveCount = this.moveCount;
    return b;
  }

  /** Build 8-channel feature tensor matching gobot_engine/board.py */
  toTensor(size) {
    // Channels:
    //  0: current player stones
    //  1: opponent stones
    //  2: empty
    //  3: current player to move = 1.0 if black else 0.0 (whole plane)
    //  4-7: previous move markers (simplified: set to 0 for demo)
    const N = size;
    const tensor = new Float32Array(8 * N * N);
    const cur = this.toMove;
    const opp = opponent(cur);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const stone = this.get(r, c);
        const base = r * N + c;
        if (stone === cur) tensor[0 * N * N + base] = 1.0;
        else if (stone === opp) tensor[1 * N * N + base] = 1.0;
        else tensor[2 * N * N + base] = 1.0;
      }
    }
    // Channel 3: to-move indicator
    const toMoveVal = (cur === BLACK) ? 1.0 : 0.0;
    for (let i = 0; i < N * N; i++) tensor[3 * N * N + i] = toMoveVal;
    // Channels 4-7 left at 0
    return tensor;
  }

  gtpToRC(coord) {
    if (!coord || coord.toLowerCase() === 'pass') return [-1, -1];
    const col = GTP_COLUMNS.indexOf(coord[0].toUpperCase());
    const row = this.size - parseInt(coord.slice(1));
    return [row, col];
  }

  rcToGtp(r, c) {
    if (r < 0) return 'pass';
    return `${GTP_COLUMNS[c]}${this.size - r}`;
  }

  legalMoves(color) {
    const moves = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (this.isLegal(r, c, color)) moves.push([r, c]);
      }
    }
    return moves;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pure-JS MCTS (fallback when model not ready, or for fast moves)
// ═══════════════════════════════════════════════════════════════════════════

class PureMCTS {
  constructor(numSims = 200) {
    this.numSims = numSims;
  }

  /** Random rollout from board state, returns +1 if startColor wins */
  rollout(board, startColor, maxMoves = 50) {
    const b = board.clone();
    let color = b.toMove;
    for (let i = 0; i < maxMoves; i++) {
      if (b.isGameOver) break;
      const moves = b.legalMoves(color);
      if (moves.length === 0) {
        b.play(-1, -1, color);
      } else {
        const [r, c] = moves[Math.floor(Math.random() * moves.length)];
        b.play(r, c, color);
      }
      color = opponent(color);
    }
    const score = b.scoreEstimate();
    const wins = (startColor === BLACK && score.startsWith('B')) ||
                 (startColor === WHITE && score.startsWith('W'));
    return wins ? 1 : -1;
  }

  getBestMove(board) {
    const color = board.toMove;
    const moves = board.legalMoves(color);
    if (moves.length === 0) return { move: [-1, -1], winrate: 0.5, candidates: [] };

    const scores = new Map();
    const visits = new Map();
    for (const [r, c] of moves) {
      scores.set(`${r},${c}`, 0);
      visits.set(`${r},${c}`, 0);
    }

    for (let s = 0; s < this.numSims; s++) {
      const [r, c] = moves[Math.floor(Math.random() * moves.length)];
      const key = `${r},${c}`;
      const b2 = board.clone();
      b2.play(r, c, color);
      const result = this.rollout(b2, color);
      scores.set(key, (scores.get(key) || 0) + result);
      visits.set(key, (visits.get(key) || 0) + 1);
    }

    let bestMove = moves[0], bestQ = -Infinity;
    const candidates = [];
    for (const [r, c] of moves) {
      const key = `${r},${c}`;
      const v = visits.get(key) || 1;
      const q = (scores.get(key) || 0) / v;
      const winrate = (q + 1) / 2;
      candidates.push({ r, c, winrate, visits: v });
      if (q > bestQ) { bestQ = q; bestMove = [r, c]; }
    }
    candidates.sort((a, b) => b.visits - a.visits);

    return {
      move: bestMove,
      winrate: (bestQ + 1) / 2,
      candidates: candidates.slice(0, 8),
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// GoEngine — ONNX Runtime Web inference + MCTS tree search
// ═══════════════════════════════════════════════════════════════════════════

class GoEngine {
  constructor() {
    this.session = null;
    this.backend = 'loading';
    this.manifest = null;
    this.pureMCTS = new PureMCTS(150);
  }

  async loadModel(modelBuffer, manifest) {
    this.manifest = manifest;

    // Try WebGPU first, then WASM
    const backends = ['webgpu', 'wasm'];
    for (const backend of backends) {
      try {
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        const session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: [backend],
        });
        this.session = session;
        this.backend = backend;
        console.log(`[GoBot] ONNX session created with backend: ${backend}`);
        return backend;
      } catch (e) {
        console.warn(`[GoBot] Backend ${backend} failed:`, e.message);
      }
    }
    throw new Error('All ONNX backends failed');
  }

  /** Run neural net inference, return { policyProbs, value } */
  async infer(board) {
    if (!this.session) return null;
    const N = board.size;
    const tensor = board.toTensor(N);
    const input = new ort.Tensor('float32', tensor, [1, 8, N, N]);
    const results = await this.session.run({ board: input });
    const logits = results['policy_logits'].data;
    const value = results['value'].data[0];

    // Softmax over policy logits
    const maxL = Math.max(...logits);
    const exp = Array.from(logits).map(x => Math.exp(x - maxL));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map(x => x / sum);

    return { probs, value };
  }

  /** MCTS-lite guided by neural network (simplified PUCT) */
  async getBestMove(board, numSims = 80) {
    const color = board.toMove;
    const legalMoves = board.legalMoves(color);
    if (legalMoves.length === 0) {
      return { move: [-1, -1], winrate: 0.5, candidates: [], tacticalOverride: null };
    }

    // Neural net evaluation of current position
    let priorProbs = null;
    let rootValue = 0.5;
    if (this.session) {
      try {
        const result = await this.infer(board);
        if (result) {
          priorProbs = result.probs;
          rootValue = (result.value + 1) / 2;  // Map [-1,1] -> [0,1]
        }
      } catch (e) {
        console.warn('[GoBot] Inference failed, falling back to uniform priors:', e.message);
      }
    }

    const N = board.size;

    // Node stats
    const Q = new Map();
    const W = new Map();
    const V = new Map();
    const P = new Map();

    for (const [r, c] of legalMoves) {
      const key = `${r},${c}`;
      Q.set(key, 0); W.set(key, 0); V.set(key, 0);
      const actionIdx = r * N + c;
      P.set(key, priorProbs ? priorProbs[actionIdx] : 1.0 / legalMoves.length);
    }

    let totalVisits = 0;
    const cPuct = 2.0;

    for (let sim = 0; sim < numSims; sim++) {
      // Select move by UCB
      let bestKey = null, bestUCB = -Infinity;
      for (const [r, c] of legalMoves) {
        const key = `${r},${c}`;
        const q = Q.get(key);
        const v = V.get(key);
        const p = P.get(key);
        const u = cPuct * p * Math.sqrt(totalVisits + 1) / (1 + v);
        const ucb = q + u;
        if (ucb > bestUCB) { bestUCB = ucb; bestKey = key; }
      }

      if (!bestKey) break;
      const [br, bc] = bestKey.split(',').map(Number);

      // Simulate
      const b2 = board.clone();
      b2.play(br, bc, color);

      let leafVal;
      if (this.session) {
        try {
          const res = await this.infer(b2);
          leafVal = res ? (res.value + 1) / 2 : 0.5;
        } catch {
          leafVal = this.pureMCTS.rollout(b2, color) > 0 ? 1 : 0;
        }
      } else {
        leafVal = this.pureMCTS.rollout(b2, color) > 0 ? 1 : 0;
      }

      // Backup
      V.set(bestKey, V.get(bestKey) + 1);
      W.set(bestKey, W.get(bestKey) + leafVal);
      Q.set(bestKey, W.get(bestKey) / V.get(bestKey));
      totalVisits++;
    }

    // Select best by visit count
    let bestMove = legalMoves[0], bestVis = 0;
    const candidates = [];
    for (const [r, c] of legalMoves) {
      const key = `${r},${c}`;
      const vis = V.get(key);
      const q = Q.get(key);
      candidates.push({ r, c, visits: vis, winrate: q, gtp: board.rcToGtp(r, c) });
      if (vis > bestVis) { bestVis = vis; bestMove = [r, c]; }
    }
    candidates.sort((a, b) => b.visits - a.visits);
    const [br, bc] = bestMove;
    const bestKey = `${br},${bc}`;

    return {
      move: bestMove,
      winrate: Q.get(bestKey) || 0.5,
      candidates: candidates.slice(0, 8),
      tacticalOverride: null,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BoardRenderer — Canvas
// ═══════════════════════════════════════════════════════════════════════════

class BoardRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.padding = 28;
  }

  starPoints(size) {
    if (size === 19) return [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]];
    if (size === 13) return [[3,3],[3,9],[6,6],[9,3],[9,9]];
    if (size === 9)  return [[2,2],[2,6],[4,4],[6,2],[6,6]];
    return [];
  }

  cellSize(size) {
    return (this.canvas.width - 2 * this.padding) / (size - 1);
  }

  coord(r, c, size) {
    const cell = this.cellSize(size);
    return [this.padding + c * cell, this.padding + r * cell];
  }

  draw(board, candidates = [], lastMove = -1) {
    const { ctx, canvas, padding } = this;
    const size = board.size;
    const cell = this.cellSize(size);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = '#d4a055';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid lines
    ctx.strokeStyle = '#7c4f1a';
    ctx.lineWidth = 1.1;
    for (let i = 0; i < size; i++) {
      ctx.beginPath();
      ctx.moveTo(padding, padding + i * cell);
      ctx.lineTo(canvas.width - padding, padding + i * cell);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding + i * cell, padding);
      ctx.lineTo(padding + i * cell, canvas.height - padding);
      ctx.stroke();
    }

    // Star points
    ctx.fillStyle = '#7c4f1a';
    for (const [r, c] of this.starPoints(size)) {
      const [x, y] = this.coord(r, c, size);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Stones
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const stone = board.get(r, c);
        if (stone !== EMPTY) this.drawStone(r, c, stone, size, cell);
      }
    }

    // Last move marker
    if (lastMove >= 0) {
      const r = Math.floor(lastMove / size);
      const c = lastMove % size;
      const stone = board.get(r, c);
      if (stone !== EMPTY) {
        const [x, y] = this.coord(r, c, size);
        ctx.strokeStyle = stone === BLACK ? '#ffffff' : '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, cell * 0.2, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Candidate overlays
    candidates.forEach((cand, idx) => {
      if (cand.r < 0) return;
      const [x, y] = this.coord(cand.r, cand.c, size);
      const alpha = Math.max(0.25, 0.85 - idx * 0.1);
      ctx.fillStyle = `rgba(56,189,248,${alpha})`;
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.36, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#0f172a';
      ctx.font = `bold ${Math.max(9, cell * 0.35)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${idx + 1}`, x, y);
    });
  }

  drawStone(r, c, color, size, cell) {
    const { ctx } = this;
    const [x, y] = this.coord(r, c, size);
    const radius = cell * 0.46;
    const grad = ctx.createRadialGradient(x - radius*0.3, y - radius*0.3, radius*0.1, x, y, radius);
    if (color === BLACK) {
      grad.addColorStop(0, '#475569');
      grad.addColorStop(1, '#090d16');
    } else {
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(1, '#cbd5e1');
    }
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  clickToRC(e, board) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const cell = this.cellSize(board.size);
    const c = Math.round((x - this.padding) / cell);
    const r = Math.round((y - this.padding) / cell);
    if (r < 0 || r >= board.size || c < 0 || c >= board.size) return null;
    return [r, c];
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// App — wires everything together
// ═══════════════════════════════════════════════════════════════════════════

class App {
  constructor() {
    this.board = new BoardState(9, 7.5);
    this.renderer = new BoardRenderer('goBoard');
    this.engine = new GoEngine();
    this.loader = new ModelLoader();
    this.candidates = [];
    this.thinking = false;
    this.modelReady = false;
    this.autoBotMove = true;

    this._setupCanvas();
    this._setupButtons();
    this._setupSizeButtons();
    this._setupToggle();
    this._render();
    this._loadModel();
  }

  _setupToggle() {
    const toggle = document.getElementById('toggleAutoMove');
    if (toggle) {
      this.autoBotMove = toggle.checked;
      toggle.addEventListener('change', (e) => {
        this.autoBotMove = e.target.checked;
        showToast(this.autoBotMove ? 'Auto bot move enabled' : 'Auto bot move disabled', 'success', 2000);
      });
    }
  }

  _setupCanvas() {
    this.renderer.canvas.addEventListener('click', (e) => {
      if (this.thinking || this.board.isGameOver) return;
      const rc = this.renderer.clickToRC(e, this.board);
      if (!rc) return;
      const [r, c] = rc;
      if (this.board.play(r, c, this.board.toMove)) {
        this.candidates = [];
        this._render();
        this._updateInfo();

        if (this.autoBotMove && !this.board.isGameOver) {
          setTimeout(() => this._botMove(), 120);
        }
      }
    });
  }

  _setupButtons() {
    document.getElementById('btnBotMove').addEventListener('click', () => this._botMove());
    document.getElementById('btnAnalyze').addEventListener('click', () => this._analyze());
    document.getElementById('btnPass').addEventListener('click', () => {
      this.board.play(-1, -1, this.board.toMove);
      this.candidates = [];
      this._render();
      this._updateInfo();

      if (this.autoBotMove && !this.board.isGameOver) {
        setTimeout(() => this._botMove(), 120);
      }
    });
    document.getElementById('btnResign').addEventListener('click', () => {
      this.board.isGameOver = true;
      this._render();
      this._updateInfo();
      showToast(`${this.board.toMove === BLACK ? 'Black' : 'White'} resigned`, 'error', 4000);
    });
    document.getElementById('btnUndo').addEventListener('click', () => {
      this.board.undo();
      this.candidates = [];
      this._render();
      this._updateInfo();
    });
    document.getElementById('btnReset').addEventListener('click', () => {
      this.board = new BoardState(this.board.size, 7.5);
      this.candidates = [];
      this._render();
      this._updateInfo();
    });
  }

  _setupSizeButtons() {
    document.querySelectorAll('.size-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const size = parseInt(btn.dataset.size);
        this.board = new BoardState(size, 7.5);
        this.candidates = [];
        // Resize canvas
        this.renderer.canvas.width = size <= 9 ? 400 : size <= 13 ? 460 : 520;
        this.renderer.canvas.height = this.renderer.canvas.width;
        this._render();
        this._updateInfo();
      });
    });
  }

  _setButtons(enabled) {
    ['btnBotMove','btnPass','btnResign','btnAnalyze','btnUndo','btnReset']
      .forEach(id => { document.getElementById(id).disabled = !enabled; });
  }

  async _loadModel() {
    const dot = document.getElementById('backendDot');
    const label = document.getElementById('backendLabel');
    dot.className = 'backend-dot loading';
    label.textContent = 'Loading model...';

    try {
      const modelBuffer = await this.loader.load((msg, type) => {
        label.textContent = msg;
        if (type === 'cache') showToast('Model loaded from cache', 'success');
      });

      const manifest = this.loader.manifest;
      const backend = await this.engine.loadModel(modelBuffer, manifest);

      this.modelReady = true;
      dot.className = `backend-dot ${backend}`;
      label.textContent = backend === 'webgpu'
        ? 'WebGPU (GPU accelerated)'
        : 'WASM (CPU)';

      const backendBadge = document.querySelector('.badge.gpu') || document.querySelector('.badge.wasm');
      // Update header badge
      const badgeEl = document.querySelector('header .badge');
      if (badgeEl) {
        badgeEl.textContent = backend === 'webgpu' ? 'WebGPU' : 'WASM';
        badgeEl.className = `badge ${backend}`;
      }

      this._setButtons(true);
      showToast(`Model ready (${backend.toUpperCase()})`, 'success');
    } catch (err) {
      dot.className = 'backend-dot';
      label.textContent = 'Model failed — using pure MCTS';
      console.error('[GoBot] Model load failed:', err);
      this._setButtons(true);
      showToast('Neural model unavailable — using pure MCTS', 'error', 5000);
    }
  }

  async _botMove() {
    if (this.thinking || this.board.isGameOver) return;
    this.thinking = true;
    this._setButtons(false);
    document.getElementById('loadingOverlay').classList.add('visible');
    document.getElementById('overlayLabel').textContent = 'Bot is thinking...';

    try {
      const result = this.modelReady
        ? await this.engine.getBestMove(this.board, 80)
        : this.engine.pureMCTS.getBestMove(this.board);

      const [r, c] = result.move;
      this.board.play(r, c, this.board.toMove);
      this.candidates = result.candidates;
      this._updateWinrate(result.winrate, this.board.toMove);
      this._renderCandidates(result.candidates);
      this._render();
      this._updateInfo();

      const tac = result.tacticalOverride;
      const tacEl = document.getElementById('tacticsAlert');
      if (tac) { tacEl.style.display = 'inline-block'; tacEl.textContent = `Tactics: ${tac}`; }
      else tacEl.style.display = 'none';
    } catch (err) {
      showToast('Bot move failed: ' + err.message, 'error');
      console.error(err);
    } finally {
      this.thinking = false;
      this._setButtons(true);
      document.getElementById('loadingOverlay').classList.remove('visible');
    }
  }

  async _analyze() {
    if (this.thinking) return;
    this.thinking = true;
    this._setButtons(false);
    document.getElementById('loadingOverlay').classList.add('visible');
    document.getElementById('overlayLabel').textContent = 'Analyzing position...';

    try {
      const result = this.modelReady
        ? await this.engine.getBestMove(this.board, 60)
        : this.engine.pureMCTS.getBestMove(this.board);

      this.candidates = result.candidates;
      this._updateWinrate(result.winrate, this.board.toMove);
      this._renderCandidates(result.candidates);
      this._render();
    } catch (err) {
      showToast('Analysis failed: ' + err.message, 'error');
    } finally {
      this.thinking = false;
      this._setButtons(true);
      document.getElementById('loadingOverlay').classList.remove('visible');
    }
  }

  _updateWinrate(winrate, toMove) {
    const bWin = Math.round((toMove === BLACK ? winrate : 1 - winrate) * 100);
    document.getElementById('blackWinrate').textContent = `${bWin}%`;
    document.getElementById('whiteWinrate').textContent = `${100 - bWin}%`;
    document.getElementById('winrateBlackFill').style.width = `${bWin}%`;
  }

  _renderCandidates(candidates) {
    const container = document.getElementById('candidatesList');
    container.innerHTML = '';
    if (!candidates.length) {
      container.innerHTML = '<span style="color:var(--text-muted);font-size:0.83rem">No candidates found.</span>';
      return;
    }
    candidates.forEach((c, i) => {
      const card = document.createElement('div');
      card.className = 'candidate-card';
      card.innerHTML = `
        <div>
          <span class="coord">#${i + 1} ${c.gtp || this.board.rcToGtp(c.r, c.c)}</span>
          <span class="visits">V: ${c.visits}</span>
        </div>
        <span class="winpct">${Math.round(c.winrate * 100)}%</span>
      `;
      container.appendChild(card);
    });
  }

  _updateInfo() {
    const b = this.board;
    const turn = b.toMove === BLACK ? 'Black ⚫' : 'White ⚪';
    document.getElementById('turnDisplay').textContent = b.isGameOver ? 'Game Over' : turn;
    document.getElementById('blackCaptures').textContent = b.captures[BLACK];
    document.getElementById('whiteCaptures').textContent = b.captures[WHITE];
    document.getElementById('scoreDisplay').textContent = b.scoreEstimate();
  }

  _render() {
    this.renderer.draw(this.board, this.candidates, this.board.lastMove);
  }
}

// Bootstrap
window.addEventListener('DOMContentLoaded', () => { window._app = new App(); });
