/**
 * Unit Tests for Pure-JS Go Board Rules & Scoring
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { Board, Color, coordToGTP, gtpToCoord } = require('../fast_mcts.js');

test('Board Initialization & Dimensions', () => {
  const b19 = new Board(19, 7.5);
  assert.equal(b19.size, 19);
  assert.equal(b19.komi, 7.5);
  assert.equal(b19.grid.length, 361);
  assert.equal(b19.to_move, Color.BLACK);

  const b9 = new Board(9, 5.5);
  assert.equal(b9.size, 9);
  assert.equal(b9.grid.length, 81);
});

test('GTP Coordinate Conversions', () => {
  // 19x19 Standard Points
  // Top-left D16: r=3, c=3
  assert.equal(coordToGTP(3, 3, 19), 'D16');
  assert.deepEqual(gtpToCoord('D16', 19), { r: 3, c: 3 });

  // Bottom-left D4: r=15, c=3
  assert.equal(coordToGTP(15, 3, 19), 'D4');
  assert.deepEqual(gtpToCoord('D4', 19), { r: 15, c: 3 });

  // Top-right Q16: r=3, c=15
  assert.equal(coordToGTP(3, 15, 19), 'Q16');
  assert.deepEqual(gtpToCoord('Q16', 19), { r: 3, c: 15 });

  // Pass move
  assert.equal(coordToGTP(-1, -1, 19), 'pass');
  assert.deepEqual(gtpToCoord('pass', 19), { r: -1, c: -1 });
  assert.deepEqual(gtpToCoord('PASS', 19), { r: -1, c: -1 });
});

test('Stone Placement & Turn Alternation', () => {
  const b = new Board(19);
  const success1 = b.play(3, 3, Color.BLACK); // D16
  assert.equal(success1, true);
  assert.equal(b.get(3, 3), Color.BLACK);
  assert.equal(b.to_move, Color.WHITE);

  const success2 = b.play(15, 15, Color.WHITE); // Q4
  assert.equal(success2, true);
  assert.equal(b.get(15, 15), Color.WHITE);
  assert.equal(b.to_move, Color.BLACK);
});

test('Single Stone Capture', () => {
  const b = new Board(9);
  // Place White stone at (4,4)
  b.set(4, 4, Color.WHITE);

  // Surround with Black stones at 4 neighbors
  b.set(3, 4, Color.BLACK);
  b.set(5, 4, Color.BLACK);
  b.set(4, 3, Color.BLACK);

  // Check legality of final capture move at (4,5)
  assert.equal(b.isLegal(4, 5, Color.BLACK), true);

  // Play final capture move
  b.to_move = Color.BLACK;
  const played = b.play(4, 5, Color.BLACK);
  assert.equal(played, true);

  // White stone at (4,4) should be removed
  assert.equal(b.get(4, 4), Color.EMPTY);
  assert.equal(b.captures[Color.BLACK], 1);
});

test('Suicide Rule Enforcement', () => {
  const b = new Board(9);
  // Surround corner (0,0) with White stones
  b.set(0, 1, Color.WHITE);
  b.set(1, 0, Color.WHITE);

  // Black playing at (0,0) with 0 liberties is suicide -> illegal
  assert.equal(b.isLegal(0, 0, Color.BLACK), false);
  const played = b.play(0, 0, Color.BLACK);
  assert.equal(played, false);
});

test('Simple Ko Rule Enforcement', () => {
  // Single-stone Ko setup:
  // Black surrounding (1,1): (0,1), (1,0), (2,1)
  // Black surrounding (1,2): (0,2), (2,2), (1,3)
  // White single stone at (1,2)
  const bKo = new Board(9);
  bKo.set(0, 1, Color.BLACK);
  bKo.set(1, 0, Color.BLACK);
  bKo.set(2, 1, Color.BLACK);

  bKo.set(0, 2, Color.BLACK);
  bKo.set(2, 2, Color.BLACK);
  bKo.set(1, 3, Color.BLACK);
  bKo.set(1, 2, Color.WHITE); // White single stone in atari at (1,2)

  // Black plays (1,1), capturing White at (1,2)
  bKo.to_move = Color.BLACK;
  const c1 = bKo.play(1, 1, Color.BLACK);
  assert.equal(c1, true);
  assert.equal(bKo.get(1, 2), Color.EMPTY); // Captured!

  // White immediately trying to recapture at (1,2) is Ko violation -> illegal
  assert.equal(bKo.isLegal(1, 2, Color.WHITE), false);
  const whiteIllegal = bKo.play(1, 2, Color.WHITE);
  assert.equal(whiteIllegal, false);
});

test('Area Scoring Flood Fill Calculation', () => {
  const b = new Board(9, 7.5);
  // Place Black wall along column 3 (c=3)
  for (let r = 0; r < 9; r++) {
    b.set(r, 3, Color.BLACK);
  }
  // Place White wall along column 5 (c=5)
  for (let r = 0; r < 9; r++) {
    b.set(r, 5, Color.WHITE);
  }

  const score = b.calculateAreaScore();
  // Columns 0, 1, 2 are Black territory (3 * 9 = 27) + 9 black stones = 36
  assert.ok(score.black_score >= 36);
  // Columns 6, 7, 8 are White territory (3 * 9 = 27) + 9 white stones + 7.5 komi = 43.5
  assert.ok(score.white_score >= 43.5);
  assert.ok(score.result_str.startsWith('W+'));
});
