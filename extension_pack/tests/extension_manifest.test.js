/**
 * Validation Tests for Extension Manifest V3 & File Integrity
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..');

test('Manifest V3 JSON Validity and Core Keys', () => {
  const manifestPath = path.join(ROOT_DIR, 'manifest.json');
  assert.ok(fs.existsSync(manifestPath), 'manifest.json must exist');

  const content = fs.readFileSync(manifestPath, 'utf8');
  const manifest = JSON.parse(content);

  assert.equal(manifest.manifest_version, 3);
  assert.ok(manifest.name);
  assert.ok(manifest.version);
  assert.ok(manifest.background && manifest.background.service_worker);
  assert.ok(manifest.action && manifest.action.default_popup);
  assert.ok(Array.isArray(manifest.content_scripts));
  assert.ok(Array.isArray(manifest.permissions));
  assert.ok(manifest.icons);
});

test('Referenced File Existence', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT_DIR, 'manifest.json'), 'utf8'));

  // Background
  const bgPath = path.join(ROOT_DIR, manifest.background.service_worker);
  assert.ok(fs.existsSync(bgPath), `Background worker missing: ${bgPath}`);

  // Popup
  const popupPath = path.join(ROOT_DIR, manifest.action.default_popup);
  assert.ok(fs.existsSync(popupPath), `Popup html missing: ${popupPath}`);

  // Icons
  for (const [size, iconRelPath] of Object.entries(manifest.icons)) {
    const iconPath = path.join(ROOT_DIR, iconRelPath);
    assert.ok(fs.existsSync(iconPath), `Icon missing for size ${size}: ${iconPath}`);
  }

  // Content Scripts & CSS
  for (const cs of manifest.content_scripts) {
    for (const jsFile of cs.js) {
      const jsPath = path.join(ROOT_DIR, jsFile);
      assert.ok(fs.existsSync(jsPath), `Content script missing: ${jsPath}`);
    }
    for (const cssFile of cs.css) {
      const cssPath = path.join(ROOT_DIR, cssFile);
      assert.ok(fs.existsSync(cssPath), `Content CSS missing: ${cssPath}`);
    }
  }
});
