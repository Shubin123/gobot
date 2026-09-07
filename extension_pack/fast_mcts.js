/**
 * GoBot Fast Pure-JavaScript Go Engine, Tactical Analyzer & MCTS
 * Full support for 19x19, 13x13, 9x9 Go board rules, liberty tracking,
 * territory estimation, Joseki book, atari tactics, and Monte Carlo Tree Search.
 * 
 * Works seamlessly in Browser Extension (Content Script, Popup, Service Worker) and Node.js.
 */

(function (global) {
  'use strict';

  // --- Constants & Color Enums ---
  const Color = {
    EMPTY: 0,
    BLACK: 1,
    WHITE: 2,
    opponent(color) {
      if (color === 1) return 2;
      if (color === 2) return 1;
      return 0;
    },
    toString(color) {
      if (color === 1) return 'B';
      if (color === 2) return 'W';
      return '.';
    },
    fromString(str) {
      if (!str) return Color.BLACK;
      const s = String(str).trim().toUpperCase();
      if (s === 'B' || s === 'BLACK' || s === '1') return Color.BLACK;
      if (s === 'W' || s === 'WHITE' || s === '2') return Color.WHITE;
      return Color.EMPTY;
    }
  };

  // Standard GTP column letters (omitting 'I')
  const GTP_COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';

  function coordToGTP(r, c, size = 19) {
    if (r === null || c === null || r < 0 || c < 0 || r >= size || c >= size) return 'pass';
    const colLetter = GTP_COLS[c];
    const rowNumber = size - r;
    return `${colLetter}${rowNumber}`;
  }

  function gtpToCoord(gtp, size = 19) {
    if (!gtp) return { r: -1, c: -1 };
    const s = String(gtp).trim().toUpperCase();
    if (s === 'PASS' || s === 'RESIGN' || s === '') return { r: -1, c: -1 };
    const colLetter = s[0];
    const c = GTP_COLS.indexOf(colLetter);
    const rowNum = parseInt(s.substring(1), 10);
    const r = size - rowNum;
    if (c < 0 || c >= size || isNaN(r) || r < 0 || r >= size) {
      return { r: -1, c: -1 };
    }
    return { r, c };
  }

  // --- Go Board Implementation ---
  class Board {
    constructor(size = 19, komi = 7.5) {
      this.size = size;
      this.komi = komi;
      this.grid = new Uint8Array(size * size);
      this.to_move = Color.BLACK;
      this.captures = { [Color.BLACK]: 0, [Color.WHITE]: 0 };
      this.ko_pos = null; // 1D index or null
      this.move_history = [];
      this.consecutive_passes = 0;
      this.is_game_over = false;
    }

    clone() {
      const b = new Board(this.size, this.komi);
      b.grid.set(this.grid);
      b.to_move = this.to_move;
      b.captures[Color.BLACK] = this.captures[Color.BLACK];
      b.captures[Color.WHITE] = this.captures[Color.WHITE];
      b.ko_pos = this.ko_pos;
      b.consecutive_passes = this.consecutive_passes;
      b.is_game_over = this.is_game_over;
      b.move_history = [...this.move_history];
      return b;
    }

    index(r, c) {
      return r * this.size + c;
    }

    coord(idx) {
      return { r: Math.floor(idx / this.size), c: idx % this.size };
    }

    inBounds(r, c) {
      return r >= 0 && r < this.size && c >= 0 && c < this.size;
    }

    get(r, c) {
      if (!this.inBounds(r, c)) return Color.EMPTY;
      return this.grid[this.index(r, c)];
    }

    set(r, c, color) {
      if (this.inBounds(r, c)) {
        this.grid[this.index(r, c)] = color;
      }
    }

    getNeighbors(r, c) {
      const neighbors = [];
      if (r > 0) neighbors.push(this.index(r - 1, c));
      if (r < this.size - 1) neighbors.push(this.index(r + 1, c));
      if (c > 0) neighbors.push(this.index(r, c - 1));
      if (c < this.size - 1) neighbors.push(this.index(r, c + 1));
      return neighbors;
    }

    getGroup(startIdx) {
      const color = this.grid[startIdx];
      if (color === Color.EMPTY) return { stones: [], liberties: [] };

      const stones = [];
      const liberties = new Set();
      const visited = new Uint8Array(this.size * this.size);
      const queue = [startIdx];
      visited[startIdx] = 1;

      while (queue.length > 0) {
        const curr = queue.pop();
        stones.push(curr);
        const { r, c } = this.coord(curr);
        const neighbors = this.getNeighbors(r, c);

        for (let i = 0; i < neighbors.length; i++) {
          const n = neighbors[i];
          const nColor = this.grid[n];
          if (nColor === Color.EMPTY) {
            liberties.add(n);
          } else if (nColor === color && !visited[n]) {
            visited[n] = 1;
            queue.push(n);
          }
        }
      }

      return { stones, liberties: Array.from(liberties) };
    }

    isLegal(r, c, color = this.to_move) {
      // Pass is always legal
      if (r === -1 && c === -1) return true;
      if (!this.inBounds(r, c)) return false;

      const idx = this.index(r, c);
      if (this.grid[idx] !== Color.EMPTY) return false;
      if (this.ko_pos === idx) return false;

      const opp = Color.opponent(color);
      const neighbors = this.getNeighbors(r, c);

      // Check if placing stone connects to direct liberties
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (this.grid[n] === Color.EMPTY) return true;
      }

      // Check if it captures any opponent groups
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (this.grid[n] === opp) {
          const oppGroup = this.getGroup(n);
          if (oppGroup.liberties.length === 1 && oppGroup.liberties[0] === idx) {
            return true; // Captures opponent group!
          }
        }
      }

      // Check if friendly neighbor group has other liberties
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (this.grid[n] === color) {
          const friendlyGroup = this.getGroup(n);
          if (friendlyGroup.liberties.length > 1) return true;
        }
      }

      // Otherwise suicide move -> illegal
      return false;
    }

    getLegalMoves(color = this.to_move) {
      const moves = [];
      for (let r = 0; r < this.size; r++) {
        for (let c = 0; c < this.size; c++) {
          if (this.isLegal(r, c, color)) {
            moves.push({ r, c });
          }
        }
      }
      moves.push({ r: -1, c: -1 }); // Pass move
      return moves;
    }

    play(r, c, color = this.to_move) {
      if (this.is_game_over) return false;

      // Handle Pass
      if (r === -1 && c === -1) {
        this.move_history.push({ r: -1, c: -1, color });
        this.consecutive_passes += 1;
        this.ko_pos = null;
        this.to_move = Color.opponent(color);
        if (this.consecutive_passes >= 2) {
          this.is_game_over = true;
        }
        return true;
      }

      if (!this.isLegal(r, c, color)) return false;

      this.consecutive_passes = 0;
      const idx = this.index(r, c);
      this.grid[idx] = color;
      const opp = Color.opponent(color);
      const neighbors = this.getNeighbors(r, c);

      let capturedStonesCount = 0;
      let singleCapturedIdx = null;

      // Capture opponent groups with 0 liberties
      const checkedGroups = new Set();
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (this.grid[n] === opp && !checkedGroups.has(n)) {
          const oppGroup = this.getGroup(n);
          for (let k = 0; k < oppGroup.stones.length; k++) {
            checkedGroups.add(oppGroup.stones[k]);
          }

          if (oppGroup.liberties.length === 0) {
            capturedStonesCount += oppGroup.stones.length;
            if (oppGroup.stones.length === 1) {
              singleCapturedIdx = oppGroup.stones[0];
            }
            for (let k = 0; k < oppGroup.stones.length; k++) {
              this.grid[oppGroup.stones[k]] = Color.EMPTY;
            }
          }
        }
      }

      this.captures[color] += capturedStonesCount;

      // Update Simple Ko status
      const selfGroup = this.getGroup(idx);
      if (capturedStonesCount === 1 && selfGroup.stones.length === 1 && selfGroup.liberties.length === 1) {
        this.ko_pos = singleCapturedIdx;
      } else {
        this.ko_pos = null;
      }

      this.move_history.push({ r, c, color });
      this.to_move = opp;
      return true;
    }

    calculateAreaScore() {
      let blackScore = 0;
      let whiteScore = this.komi;
      const visited = new Uint8Array(this.size * this.size);

      for (let i = 0; i < this.grid.length; i++) {
        if (this.grid[i] === Color.BLACK) {
          blackScore += 1;
        } else if (this.grid[i] === Color.WHITE) {
          whiteScore += 1;
        } else if (!visited[i]) {
          // Empty territory flood-fill
          const territory = [];
          const queue = [i];
          visited[i] = 1;
          let touchesBlack = false;
          let touchesWhite = false;

          while (queue.length > 0) {
            const curr = queue.pop();
            territory.push(curr);
            const { r, c } = this.coord(curr);
            const neighbors = this.getNeighbors(r, c);

            for (let k = 0; k < neighbors.length; k++) {
              const n = neighbors[k];
              const col = this.grid[n];
              if (col === Color.BLACK) touchesBlack = true;
              else if (col === Color.WHITE) touchesWhite = true;
              else if (col === Color.EMPTY && !visited[n]) {
                visited[n] = 1;
                queue.push(n);
              }
            }
          }

          if (touchesBlack && !touchesWhite) {
            blackScore += territory.length;
          } else if (touchesWhite && !touchesBlack) {
            whiteScore += territory.length;
          }
        }
      }

      const diff = blackScore - whiteScore;
      let resultStr = '0.0';
      if (diff > 0) {
        resultStr = `B+${diff.toFixed(1)}`;
      } else if (diff < 0) {
        resultStr = `W+${Math.abs(diff).toFixed(1)}`;
      } else {
        resultStr = 'Draw';
      }

      return {
        black_score: blackScore,
        white_score: whiteScore,
        diff,
        result_str: resultStr
      };
    }
  }

  // --- Tactical Analyzer & Joseki Book ---
  class TacticalAnalyzer {
    constructor() {
      // 19x19 Joseki and opening star patterns
      this.starPoints19 = [
        { r: 3, c: 3, gtp: 'D16', desc: 'Top-Left Star Point (Hoshi)' },
        { r: 3, c: 15, gtp: 'Q16', desc: 'Top-Right Star Point (Hoshi)' },
        { r: 15, c: 3, gtp: 'D4', desc: 'Bottom-Left Star Point (Hoshi)' },
        { r: 15, c: 15, gtp: 'Q4', desc: 'Bottom-Right Star Point (Hoshi)' },
        { r: 3, c: 9, gtp: 'K16', desc: 'Top Side Star Point' },
        { r: 15, c: 9, gtp: 'K4', desc: 'Bottom Side Star Point' },
        { r: 9, c: 3, gtp: 'D10', desc: 'Left Side Star Point' },
        { r: 9, c: 15, gtp: 'Q10', desc: 'Right Side Star Point' },
        { r: 9, c: 9, gtp: 'K10', desc: 'Tengen (Center Star Point)' }
      ];

      // Standard Joseki Approaches & Responses
      this.josekiPatterns19 = [
        // 3-3 Corner Invasions
        { r: 2, c: 2, gtp: 'C17', desc: '3-3 Corner Invasion (San-san)' },
        { r: 2, c: 16, gtp: 'R17', desc: '3-3 Corner Invasion (San-san)' },
        { r: 16, c: 2, gtp: 'C3', desc: '3-3 Corner Invasion (San-san)' },
        { r: 16, c: 16, gtp: 'R3', desc: '3-3 Corner Invasion (San-san)' },
        // Knight Approaches
        { r: 2, c: 14, gtp: 'P17', desc: 'Knight Approach (Kakari)' },
        { r: 4, c: 16, gtp: 'R15', desc: 'Knight Approach (Kakari)' },
        { r: 16, c: 14, gtp: 'P3', desc: 'Knight Approach (Kakari)' },
        { r: 14, c: 16, gtp: 'R5', desc: 'Knight Approach (Kakari)' },
        { r: 2, c: 4, gtp: 'E17', desc: 'Knight Approach (Kakari)' },
        { r: 4, c: 2, gtp: 'C15', desc: 'Knight Approach (Kakari)' },
        { r: 16, c: 4, gtp: 'E3', desc: 'Knight Approach (Kakari)' },
        { r: 14, c: 2, gtp: 'C5', desc: 'Knight Approach (Kakari)' },
        // 3-4 Komoku Points
        { r: 2, c: 3, gtp: 'D17', desc: '3-4 Point (Komoku)' },
        { r: 3, c: 2, gtp: 'C16', desc: '3-4 Point (Komoku)' },
        { r: 2, c: 15, gtp: 'Q17', desc: '3-4 Point (Komoku)' },
        { r: 3, c: 16, gtp: 'R16', desc: '3-4 Point (Komoku)' },
        { r: 16, c: 3, gtp: 'D3', desc: '3-4 Point (Komoku)' },
        { r: 15, c: 2, gtp: 'C4', desc: '3-4 Point (Komoku)' },
        { r: 16, c: 15, gtp: 'Q3', desc: '3-4 Point (Komoku)' },
        { r: 15, c: 16, gtp: 'R4', desc: '3-4 Point (Komoku)' }
      ];
    }

    findTacticalMoves(board, color) {
      const opp = Color.opponent(color);
      const tacticalMoves = [];
      const checkedGroups = new Set();

      // 1. Immediate Capture Moves (Enemy in Atari)
      for (let i = 0; i < board.grid.length; i++) {
        if (board.grid[i] === opp && !checkedGroups.has(i)) {
          const group = board.getGroup(i);
          group.stones.forEach(s => checkedGroups.add(s));

          if (group.liberties.length === 1) {
            const libIdx = group.liberties[0];
            const coord = board.coord(libIdx);
            if (board.isLegal(coord.r, coord.c, color)) {
              tacticalMoves.push({
                r: coord.r,
                c: coord.c,
                gtp: coordToGTP(coord.r, coord.c, board.size),
                type: 'ATARI_CAPTURE',
                priority: 95,
                desc: `Atari Capture (${group.stones.length} stones)`
              });
            }
          }
        }
      }

      // 2. Urgent Escape / Atari Defense Moves (Friendly in Atari)
      const friendlyChecked = new Set();
      for (let i = 0; i < board.grid.length; i++) {
        if (board.grid[i] === color && !friendlyChecked.has(i)) {
          const group = board.getGroup(i);
          group.stones.forEach(s => friendlyChecked.add(s));

          if (group.liberties.length === 1) {
            const libIdx = group.liberties[0];
            const coord = board.coord(libIdx);
            if (board.isLegal(coord.r, coord.c, color)) {
              tacticalMoves.push({
                r: coord.r,
                c: coord.c,
                gtp: coordToGTP(coord.r, coord.c, board.size),
                type: 'ATARI_DEFENSE',
                priority: 90 + Math.min(group.stones.length, 9),
                desc: `Atari Defense / Escape (${group.stones.length} stones)`
              });
            }
          }
        }
      }

      // 3. Opening Star & Joseki Book (for 19x19 early game)
      if (board.size === 19 && board.move_history.length < 30) {
        // Star points
        for (const sp of this.starPoints19) {
          if (board.get(sp.r, sp.c) === Color.EMPTY && board.isLegal(sp.r, sp.c, color)) {
            tacticalMoves.push({
              r: sp.r,
              c: sp.c,
              gtp: sp.gtp,
              type: 'OPENING_STAR',
              priority: 70,
              desc: sp.desc
            });
          }
        }
        // Joseki / Approaches
        for (const jp of this.josekiPatterns19) {
          if (board.get(jp.r, jp.c) === Color.EMPTY && board.isLegal(jp.r, jp.c, color)) {
            tacticalMoves.push({
              r: jp.r,
              c: jp.c,
              gtp: jp.gtp,
              type: 'JOSEKI_PATTERN',
              priority: 65,
              desc: jp.desc
            });
          }
        }
      } else if (board.size === 9 && board.move_history.length < 8) {
        // 9x9 Star and Center Points
        const pts9 = [
          { r: 4, c: 4, gtp: 'E5', desc: 'Tengen (Center Focus)' },
          { r: 2, c: 2, gtp: 'C7', desc: '3-3 Corner Point' },
          { r: 2, c: 6, gtp: 'G7', desc: '3-3 Corner Point' },
          { r: 6, c: 2, gtp: 'C3', desc: '3-3 Corner Point' },
          { r: 6, c: 6, gtp: 'G3', desc: '3-3 Corner Point' }
        ];
        for (const p of pts9) {
          if (board.get(p.r, p.c) === Color.EMPTY && board.isLegal(p.r, p.c, color)) {
            tacticalMoves.push({
              r: p.r,
              c: p.c,
              gtp: p.gtp,
              type: 'OPENING_9x9',
              priority: 75,
              desc: p.desc
            });
          }
        }
      }

      return tacticalMoves.sort((a, b) => b.priority - a.priority);
    }
  }

  // --- Fast Monte Carlo Tree Search (MCTS) Engine ---
  class MCTSNode {
    constructor(board, parent = null, move = null, prior = 1.0) {
      this.board = board;
      this.parent = parent;
      this.move = move; // { r, c }
      this.prior = prior;
      this.visits = 0;
      this.totalValue = 0;
      this.children = new Map(); // key `${r},${c}` -> MCTSNode
      this.isExpanded = false;
    }

    get winrate() {
      return this.visits === 0 ? 0.5 : this.totalValue / this.visits;
    }

    get ucbScore() {
      const c_puct = 2.0;
      if (this.visits === 0) {
        const parentVisits = this.parent ? this.parent.visits : 1;
        return 0.5 + c_puct * this.prior * (Math.sqrt(parentVisits) / (1 + this.visits));
      }
      const parentVisits = this.parent ? this.parent.visits : this.visits;
      const q = this.totalValue / this.visits;
      const u = c_puct * this.prior * (Math.sqrt(parentVisits) / (1 + this.visits));
      return q + u;
    }
  }

  class FastMCTS {
    constructor(options = {}) {
      this.numSimulations = options.numSimulations || 60;
      this.cPuct = options.cPuct || 2.0;
      this.tacticalAnalyzer = new TacticalAnalyzer();
      this.enableTactics = options.enableTactics !== false;
    }

    staticEvaluate(board, perspectiveColor) {
      const score = board.calculateAreaScore();
      const opp = Color.opponent(perspectiveColor);
      const myCaptures = board.captures[perspectiveColor];
      const oppCaptures = board.captures[opp];

      let rawVal = 0.5;
      if (perspectiveColor === Color.BLACK) {
        rawVal = 0.5 + (score.diff / (board.size * board.size * 0.4));
      } else {
        rawVal = 0.5 - (score.diff / (board.size * board.size * 0.4));
      }

      rawVal += (myCaptures - oppCaptures) * 0.03;
      return Math.max(0.01, Math.min(0.99, rawVal));
    }

    getHeuristicPrior(board, r, c, color, tacticalMap) {
      if (r === -1 && c === -1) return 0.05; // Pass prior

      const key = `${r},${c}`;
      if (tacticalMap && tacticalMap.has(key)) {
        const tac = tacticalMap.get(key);
        return 0.3 + (tac.priority / 100.0) * 0.6;
      }

      // Proximity to existing stones and star points
      let score = 0.1;
      const distToCenter = Math.abs(r - Math.floor(board.size / 2)) + Math.abs(c - Math.floor(board.size / 2));
      score += Math.max(0, 0.15 - (distToCenter / (board.size * 2)));

      // Encourage 3rd and 4th lines for territory & balance
      const rDist = Math.min(r, board.size - 1 - r);
      const cDist = Math.min(c, board.size - 1 - c);
      if ((rDist === 2 || rDist === 3) && (cDist === 2 || cDist === 3)) {
        score += 0.2;
      }

      // Penalty for 1st line (edge) in early game
      if (rDist === 0 || cDist === 0) {
        score -= 0.08;
      }

      return Math.max(0.02, Math.min(0.95, score));
    }

    search(board, simulations = this.numSimulations) {
      const rootColor = board.to_move;
      const root = new MCTSNode(board.clone());

      // Tactical Pre-scan
      const tacticalMoves = this.tacticalAnalyzer.findTacticalMoves(board, rootColor);
      const tacticalMap = new Map();
      tacticalMoves.forEach(m => tacticalMap.set(`${m.r},${m.c}`, m));

      // Quick return if high priority tactical move exists (atari defense / capture)
      if (this.enableTactics && tacticalMoves.length > 0 && tacticalMoves[0].priority >= 90) {
        const topTac = tacticalMoves[0];
        const winrateEst = topTac.type === 'ATARI_CAPTURE' ? 0.72 : 0.61;
        return {
          selected_move: topTac.gtp,
          row: topTac.r,
          col: topTac.c,
          winrate: winrateEst,
          tactical_override: true,
          top_candidates: tacticalMoves.slice(0, 8).map(m => ({
            move: m.gtp,
            row: m.r,
            col: m.c,
            winrate: m.type === 'ATARI_CAPTURE' ? 0.72 : 0.61,
            visits: 100,
            policy_prior: 0.85,
            description: m.desc
          }))
        };
      }

      // Perform MCTS Simulations
      for (let sim = 0; sim < simulations; sim++) {
        let node = root;
        const searchBoard = board.clone();

        // 1. Selection
        while (node.isExpanded && node.children.size > 0) {
          let bestScore = -Infinity;
          let bestChild = null;

          for (const child of node.children.values()) {
            const score = child.ucbScore;
            if (score > bestScore) {
              bestScore = score;
              bestChild = child;
            }
          }

          if (!bestChild) break;
          node = bestChild;
          if (node.move) {
            searchBoard.play(node.move.r, node.move.c, searchBoard.to_move);
          }
        }

        // 2. Expansion
        if (!node.isExpanded && !searchBoard.is_game_over) {
          const legalMoves = searchBoard.getLegalMoves();
          let priorSum = 0;
          const priors = [];

          for (const m of legalMoves) {
            const p = this.getHeuristicPrior(searchBoard, m.r, m.c, searchBoard.to_move, tacticalMap);
            priors.push(p);
            priorSum += p;
          }

          for (let i = 0; i < legalMoves.length; i++) {
            const m = legalMoves[i];
            const normalizedPrior = priorSum > 0 ? priors[i] / priorSum : 1.0 / legalMoves.length;
            const childNode = new MCTSNode(searchBoard, node, m, normalizedPrior);
            node.children.set(`${m.r},${m.c}`, childNode);
          }
          node.isExpanded = true;
        }

        // 3. Static Evaluation
        const evalValue = this.staticEvaluate(searchBoard, rootColor);

        // 4. Backpropagation
        let curr = node;
        while (curr !== null) {
          curr.visits += 1;
          curr.totalValue += evalValue;
          curr = curr.parent;
        }
      }

      // Collect candidates
      const candidates = [];
      for (const [key, child] of root.children.entries()) {
        const move = child.move;
        const gtp = coordToGTP(move.r, move.c, board.size);
        const tac = tacticalMap.get(`${move.r},${move.c}`);
        const desc = tac ? tac.desc : (gtp === 'pass' ? 'Pass turn' : `Positional Move ${gtp}`);

        candidates.push({
          move: gtp,
          row: move.r >= 0 ? move.r : null,
          col: move.c >= 0 ? move.c : null,
          winrate: Math.round(child.winrate * 1000) / 1000,
          visits: child.visits,
          policy_prior: Math.round(child.prior * 1000) / 1000,
          description: desc
        });
      }

      candidates.sort((a, b) => b.visits - a.visits || b.winrate - a.winrate);

      const best = candidates.length > 0 ? candidates[0] : {
        move: 'pass',
        row: null,
        col: null,
        winrate: 0.5,
        visits: 1,
        policy_prior: 0.1,
        description: 'No legal moves, pass.'
      };

      return {
        selected_move: best.move,
        row: best.row,
        col: best.col,
        winrate: best.winrate,
        tactical_override: false,
        top_candidates: candidates.slice(0, 8)
      };
    }
  }

  // --- Export Module ---
  const GoBotFastEngine = {
    Color,
    Board,
    GTP_COLS,
    coordToGTP,
    gtpToCoord,
    TacticalAnalyzer,
    FastMCTS
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = GoBotFastEngine;
  }
  if (typeof globalThis !== 'undefined') {
    globalThis.GoBotFastEngine = GoBotFastEngine;
  }
})(typeof self !== 'undefined' ? self : this);
