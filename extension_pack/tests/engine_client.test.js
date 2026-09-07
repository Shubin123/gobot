/**
 * Unit Tests for GoBot Engine Client & Fallback Mechanisms
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { GoBotEngineClient } = require('../engine_client.js');

test('Engine Client - Fallback to Pure JS when Offline', async () => {
  const client = new GoBotEngineClient({
    serverUrl: 'http://127.0.0.1:59999', // Non-existent offline port
    numSimulations: 30,
    timeoutMs: 300
  });

  const health = await client.checkHealth();
  assert.equal(health.mode, 'pure_js_fallback');
  assert.equal(health.status, 'ok');

  const opt = await client.optimizeMove({
    board_size: 19,
    to_move: 'B',
    num_simulations: 20
  });

  assert.equal(opt.engine_backend, 'pure_js_fallback');
  assert.ok(opt.selected_move);
  assert.ok(opt.top_candidates.length > 0);
  assert.ok(opt.winrate >= 0 && opt.winrate <= 1.0);
});

test('Engine Client - Board Analyzer Evaluation', async () => {
  const client = new GoBotEngineClient({
    forcePureJS: true,
    numSimulations: 25
  });

  const analysis = await client.analyzeBoard({
    board_size: 9,
    to_move: 'B',
    num_simulations: 20
  });

  assert.equal(analysis.engine_backend, 'pure_js_fallback');
  assert.ok(analysis.current_score_estimate);
  assert.ok(analysis.candidates.length > 0);
});
