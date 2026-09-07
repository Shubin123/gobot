/**
 * Unit Tests for Tactical Analyzer and Fast MCTS Engine
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { Board, Color, TacticalAnalyzer, FastMCTS } = require('../fast_mcts.js');

test('Tactical Analyzer - Atari Capture Detection', () => {
  const b = new Board(19);
  const analyzer = new TacticalAnalyzer();

  // White stone at (3,3) with 3 neighbors blocked by Black, 1 liberty at (3,4)
  b.set(3, 3, Color.WHITE);
  b.set(2, 3, Color.BLACK);
  b.set(4, 3, Color.BLACK);
  b.set(3, 2, Color.BLACK);
  // (3,4) is Empty -> White is in Atari!

  const tactics = analyzer.findTacticalMoves(b, Color.BLACK);
  assert.ok(tactics.length > 0);

  const captureMove = tactics.find(t => t.type === 'ATARI_CAPTURE');
  assert.ok(captureMove);
  assert.equal(captureMove.r, 3);
  assert.equal(captureMove.c, 4);
  assert.equal(captureMove.gtp, 'E16');
});

test('Tactical Analyzer - Atari Defense Detection', () => {
  const b = new Board(19);
  const analyzer = new TacticalAnalyzer();

  // Black stones at (15,3) in Atari
  b.set(15, 3, Color.BLACK);
  b.set(14, 3, Color.WHITE);
  b.set(16, 3, Color.WHITE);
  b.set(15, 2, Color.WHITE);
  // (15,4) is Empty -> Black in Atari!

  const tactics = analyzer.findTacticalMoves(b, Color.BLACK);
  const defenseMove = tactics.find(t => t.type === 'ATARI_DEFENSE');
  assert.ok(defenseMove);
  assert.equal(defenseMove.r, 15);
  assert.equal(defenseMove.c, 4);
  assert.equal(defenseMove.gtp, 'E4');
});

test('Tactical Analyzer - 19x19 Opening Star & Joseki Patterns', () => {
  const b = new Board(19);
  const analyzer = new TacticalAnalyzer();

  const openingMoves = analyzer.findTacticalMoves(b, Color.BLACK);
  assert.ok(openingMoves.length >= 8);

  const starMoves = openingMoves.filter(m => m.type === 'OPENING_STAR');
  assert.ok(starMoves.some(m => m.gtp === 'D16' || m.gtp === 'Q16' || m.gtp === 'D4' || m.gtp === 'Q4'));
});

test('Fast MCTS Search Output Contract & Validity', () => {
  const b = new Board(19);
  const mcts = new FastMCTS({ numSimulations: 40, enableTactics: true });

  const result = mcts.search(b, 40);

  assert.ok(result.selected_move);
  assert.notEqual(result.row, undefined);
  assert.notEqual(result.col, undefined);
  assert.ok(typeof result.winrate === 'number');
  assert.ok(result.winrate >= 0 && result.winrate <= 1.0);
  assert.ok(Array.isArray(result.top_candidates));
  assert.ok(result.top_candidates.length > 0);

  // Validate top candidate structure
  const top1 = result.top_candidates[0];
  assert.ok(top1.move);
  assert.ok(top1.winrate !== undefined);
  assert.ok(top1.visits !== undefined);
  assert.ok(top1.policy_prior !== undefined);
  assert.ok(top1.description);
});

test('Fast MCTS Tactical Override on Urgent Atari', () => {
  const b = new Board(19);
  // White group in Atari
  b.set(5, 5, Color.WHITE);
  b.set(4, 5, Color.BLACK);
  b.set(6, 5, Color.BLACK);
  b.set(5, 4, Color.BLACK);
  b.to_move = Color.BLACK;

  const mcts = new FastMCTS({ numSimulations: 30, enableTactics: true });
  const result = mcts.search(b, 30);

  // Must execute immediate tactical capture at (5,6)
  assert.equal(result.tactical_override, true);
  assert.equal(result.row, 5);
  assert.equal(result.col, 6);
});
