# GoBot - Neural Go Assistant Browser Extension (Manifest V3)

A high-performance, production-ready JavaScript Browser Extension for real-time Go / Baduk / Weiqi game analysis. Compatible with **Chrome**, **Edge**, **Brave**, **Opera**, and **Firefox**.

---

## 🌟 Features

### 1. Universal Live Board Detection & Overlays
- **Supported Platforms**: Online-Go.com (OGS), Fox Go Web, BadukPop, KGS Web, Sabaki Web, Yike Weiqi, and custom HTML5 canvas/SVG boards.
- **Dynamic Candidate Heatmap**: Visualizes move priorities with color-coded gradients (Emerald `#1`, Blue `#2`, Purple `#3`, Amber tactical defenses).
- **Optimal Move Halo**: Pulsing glow around the engine's highest-confidence suggestion.
- **Ghost Stone Hover Previews**: Hover over candidate chips to inspect potential territory impact.
- **Floating HUD Overlay**: Glassmorphic draggable in-page widget with real-time Black vs White winrate bar, top moves list, and mode toggles.

### 2. Dual Engine Architecture (FastAPI + Pure-JS Fallback)
- **FastAPI Neural Engine**: Connects to your local GoBot Python backend (`http://localhost:8000`) utilizing PyTorch `GoResNet` and deep MCTS.
- **Seamless Pure-JS Fallback**: If the Python server is offline or unavailable, the extension automatically transitions to an embedded pure-JavaScript Monte Carlo Tree Search (`fast_mcts.js`) with:
  - Full Go rules (19x19, 13x13, 9x9): liberty tracking, flood-fill capture, suicide prevention, simple Ko enforcement.
  - Tactical heuristics: immediate Atari capture/defense, cut/connection protection.
  - Joseki opening book: Star points (4-4), 3-3 corner invasions, knight approaches, 3-4 komoku points.
  - Fast area scoring & territory flood fill.

### 3. Rich Popup Studio UI
- **Live Advisor**: Displays real-time winrate percentage, territory lead estimate, and candidate moves with tactical tags.
- **Board Studio**: Interactive 19x19 / 13x13 / 9x9 board with stone placement tools, star point presets, and manual analysis.
- **Settings**: Configure simulation budget (10–250+ sims), exploration constant ($c_{puct}$), backend server URL, and force offline mode.

---

## 🚀 Installation Guide

### Load Unpacked in Chrome / Edge / Brave
1. Open your browser and navigate to the Extensions page:
   - Chrome / Brave: `chrome://extensions/`
   - Edge: `edge://extensions/`
2. Enable **Developer mode** (toggle in top-right corner).
3. Click **Load unpacked**.
4. Select the `dist/unpacked/` directory inside `extension_pack/` (or run `npm run build` first).

### Load Temporary Add-on in Firefox
1. Navigate to `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on...**.
3. Select `extension_pack/dist/unpacked/manifest.json` or `extension_pack/manifest.json`.

---

## 🛠️ Build & Packaging Scripts

All packaging and testing scripts are self-contained with **zero external npm dependencies** (using Node.js standard built-ins).

```bash
# Run unit tests
npm test

# Build distribution in dist/unpacked and create zip package
npm run build

# Generate PNG and SVG icons
npm run generate-icons
```

---

## 📂 Project Structure

```
extension_pack/
├── manifest.json            # Manifest V3 configuration
├── background.js           # Service worker (tab routing, badge state, context menu)
├── content_script.js       # In-page board detector, heatmap & floating HUD renderer
├── overlay.css             # Glassmorphic HUD and candidate overlay styling
├── engine_client.js        # FastAPI bridge with auto-fallback to pure-JS MCTS
├── fast_mcts.js            # Pure-JS Go rules, tactical analyzer & MCTS engine
├── package.json            # NPM build & test scripts
├── build.js                # Packaging & ZIP creation script
├── icons/
│   ├── icon16.png
│   ├── icon32.png
│   ├── icon48.png
│   ├── icon128.png
│   ├── icon.svg
│   └── generate_icons.js
├── popup/
│   ├── popup.html          # Advisor, Board Studio & Settings tabs
│   ├── popup.css           # Dark-mode theme & metrics styling
│   └── popup.js            # Interactive board & extension controller
├── tests/
│   ├── board_rules.test.js
│   ├── fast_mcts.test.js
│   ├── engine_client.test.js
│   └── extension_manifest.test.js
└── dist/
    ├── unpacked/           # Ready-to-load unpacked browser extension
    └── gobot-extension.zip # Production distribution ZIP package
```
