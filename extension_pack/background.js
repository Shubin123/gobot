/**
 * GoBot Extension - Manifest V3 Background Service Worker
 * Coordinates engine requests, tab state, context menus, settings storage, and status badges.
 */

// Import bundled engine modules
importScripts('fast_mcts.js', 'engine_client.js');

const DEFAULT_SETTINGS = {
  serverUrl: 'http://localhost:8000',
  numSimulations: 60,
  enableTactics: true,
  forcePureJS: false,
  gameMode: 'advisor', // 'advisor' | 'bot_assist'
  showHeatmap: true,
  showWinrateBar: true,
  autoAnalyze: true,
  boardTheme: 'cyber_zen'
};

let engineClient = new GoBotEngineClient(DEFAULT_SETTINGS);

// Load persisted settings on startup
chrome.storage.local.get(DEFAULT_SETTINGS, (stored) => {
  const settings = { ...DEFAULT_SETTINGS, ...stored };
  engineClient = new GoBotEngineClient(settings);
  updateBadgeStatus();
});

// Update extension icon badge
async function updateBadgeStatus() {
  try {
    const health = await engineClient.checkHealth();
    if (health.mode === 'fastapi') {
      chrome.action.setBadgeText({ text: 'AI' });
      chrome.action.setBadgeBackgroundColor({ color: '#10B981' }); // Emerald Green
      chrome.action.setTitle({ title: 'GoBot: FastAPI Server Connected' });
    } else {
      chrome.action.setBadgeText({ text: 'JS' });
      chrome.action.setBadgeBackgroundColor({ color: '#3B82F6' }); // Blue
      chrome.action.setTitle({ title: 'GoBot: Pure JS Fallback Engine Active' });
    }
  } catch (err) {
    chrome.action.setBadgeText({ text: 'JS' });
    chrome.action.setBadgeBackgroundColor({ color: '#6B7280' });
  }
}

// Initial badge check & interval
chrome.runtime.onInstalled.addListener(() => {
  updateBadgeStatus();
  createContextMenus();
});

// Create context menus for in-page quick analysis
function createContextMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: 'gobot-analyze-page',
      title: 'GoBot: Analyze Online Board',
      contexts: ['all']
    });
    chrome.contextMenus.create({
      id: 'gobot-toggle-hud',
      title: 'GoBot: Toggle Live Overlay HUD',
      contexts: ['all']
    });
  });
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab || !tab.id) return;
  if (info.menuItemId === 'gobot-analyze-page') {
    chrome.tabs.sendMessage(tab.id, { action: 'TRIGGER_MANUAL_ANALYZE' });
  } else if (info.menuItemId === 'gobot-toggle-hud') {
    chrome.tabs.sendMessage(tab.id, { action: 'TOGGLE_OVERLAY_HUD' });
  }
});

// Handle incoming messages from Content Script and Popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const action = request.action;

  if (action === 'PING_ENGINE') {
    engineClient.checkHealth().then(health => {
      updateBadgeStatus();
      sendResponse({ ok: true, health });
    }).catch(err => {
      sendResponse({ ok: false, error: err.message });
    });
    return true; // async
  }

  if (action === 'OPTIMIZE_MOVE') {
    chrome.action.setBadgeText({ text: '...' });
    chrome.action.setBadgeBackgroundColor({ color: '#F59E0B' }); // Amber computing

    engineClient.optimizeMove(request.params || {}).then(result => {
      updateBadgeStatus();
      sendResponse({ ok: true, result });
    }).catch(err => {
      updateBadgeStatus();
      sendResponse({ ok: false, error: err.message });
    });
    return true; // async
  }

  if (action === 'ANALYZE_BOARD') {
    engineClient.analyzeBoard(request.params || {}).then(result => {
      sendResponse({ ok: true, result });
    }).catch(err => {
      sendResponse({ ok: false, error: err.message });
    });
    return true; // async
  }

  if (action === 'GET_SETTINGS') {
    chrome.storage.local.get(DEFAULT_SETTINGS, (settings) => {
      sendResponse({ ok: true, settings });
    });
    return true;
  }

  if (action === 'SAVE_SETTINGS') {
    chrome.storage.local.set(request.settings, () => {
      chrome.storage.local.get(DEFAULT_SETTINGS, (updated) => {
        engineClient = new GoBotEngineClient(updated);
        updateBadgeStatus();
        sendResponse({ ok: true, settings: updated });
      });
    });
    return true;
  }

  if (action === 'FORWARD_TO_TAB') {
    if (sender.tab && sender.tab.id) {
      chrome.tabs.sendMessage(sender.tab.id, request.payload, sendResponse);
      return true;
    }
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, request.payload, sendResponse);
      } else {
        sendResponse({ ok: false, error: 'No active tab found' });
      }
    });
    return true;
  }

  return false;
});
