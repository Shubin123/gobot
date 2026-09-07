# GoBot — Dual-Headed ResNet Go Engine

[![CI](https://github.com/Shubin123/gobot/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubin123/gobot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)

> A production-grade Go AI engine: dual-headed ResNet policy+value network, PUCT MCTS, GTP 2.0, tactical solvers, curriculum training, self-play, and a live browser demo.

**🎮 [Play the live demo →](https://shubin123.github.io/gobot)**  
*(Runs the neural network entirely in your browser via WebGPU / WASM — no server required)*

---

## Features

- **Neural Network** — Dual-headed ResNet with policy + value heads (AlphaGo Zero style)
- **MCTS** — PUCT formula, Dirichlet noise, configurable simulations
- **Tactical Engine** — Ladder solving, atari capture, eye detection, joseki opening book
- **GTP 2.0** — Compatible with Sabaki, Lizzie, Katrain, GoGui
- **Training Pipeline** — SGF parsing, curriculum filtering by rank/phase, dihedral augmentation, self-play
- **FastAPI Server** — REST + WebSocket game API with web UI
- **Browser Extension** — Chrome MV3 extension for OGS, Fox, KGS (overlay heatmaps + winrate)
- **GH Pages Demo** — GPU-accelerated inference via ONNX Runtime Web (WebGPU backend)

---

## Quick Start

```bash
# Install
pip install -e .

# Play interactively in terminal (9×9)
gobot play --board-size 9

# Run GTP engine (for Sabaki / GoGui)
gobot gtp --board-size 19 --model checkpoints/winning_gobot_model.pt

# Start web server + UI
gobot server --port 8000
# → open http://localhost:8000

# Train on SGF files
gobot train --sources /path/to/sgfs/ --board-size 9 --epochs 5

# Self-play data generation
gobot selfplay --num-games 10 --board-size 9
```

---

## Project Structure

```
gobot/
├── gobot_engine/          # Core engine
│   ├── board.py           #   Go rules, captures, Ko, scoring
│   ├── neural_net.py      #   Dual-head ResNet (policy + value)
│   ├── mcts.py            #   PUCT MCTS with neural guidance
│   ├── optimizer.py       #   Move optimizer + tactical heuristics
│   ├── ladder.py          #   Ladder solver
│   ├── joseki.py          #   Opening book
│   └── gtp.py             #   GTP 2.0 engine
├── training_pipeline/     # Training
│   ├── sgf_parser.py      #   SGF file parser
│   ├── dataset.py         #   GoDataset + dihedral augmentation
│   ├── curriculum.py      #   Curriculum filtering
│   ├── trainer.py         #   Training loop
│   ├── self_play.py       #   Self-play data generator
│   └── auto_trainer.py    #   Automated modulated training
├── online_extension/      # Web server + UI
│   ├── server.py          #   FastAPI REST + WebSocket
│   └── web_ui/            #   Canvas board UI
├── extension_pack/        # Chrome MV3 extension (OGS, Fox, KGS)
├── docs/                  # GitHub Pages demo (ONNX + WebGPU)
├── scripts/
│   └── export_onnx.py     # PyTorch → ONNX export + shard tool
├── tests/                 # Full test suite
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_module.py
│   ├── test_integration.py
│   ├── test_capabilities.py
│   ├── test_training.py
│   ├── test_e2e.py
│   └── e2e/
│       └── test_full_game.py
├── checkpoints/           # Model weights
└── cli.py                 # CLI entry point
```

---

## Model Architecture

```
Input: (B, 8, N, N)   — 8-channel board representation
  → ConvBlock (initial)
  → 6× ResBlock (residual tower)
  ├─ Policy Head → softmax → (N*N + 1) move probabilities
  └─ Value Head  → tanh    → scalar win probability in [-1, 1]
```

Default: 64 filters, 6 residual blocks, board sizes 9/13/19.

---

## Browser Demo (GH Pages)

The live demo exports the trained model to ONNX and runs it with:
- **WebGPU backend** for GPU-accelerated inference (Chrome 113+, Edge)
- **WASM fallback** for all other browsers
- **IndexedDB caching** — model downloads once (~1.5 MB), cached locally forever
- Download progress banner with size warning on first visit
- Pure client-side — no server, no telemetry

---

## Browser Extension

Load `extension_pack/` as an unpacked Chrome extension:
1. Navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. "Load unpacked" → select `extension_pack/`

Supports: OGS, Fox, KGS, BadukPop, Sabaki (localhost)

---

## Running Tests

```bash
pip install -e ".[dev]"

# All tests
pytest

# Only smoke tests (fast, < 2s each)
pytest -m smoke

# Integration tests
pytest -m integration

# End-to-end
pytest tests/e2e/

# With coverage
pytest --cov=gobot_engine --cov=training_pipeline --cov-report=html
```

---

## License

MIT — see [LICENSE](LICENSE)
