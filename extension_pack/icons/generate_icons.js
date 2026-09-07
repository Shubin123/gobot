/**
 * Pure Node.js PNG and SVG Icon Generator for GoBot Extension
 * Generates crisp 16x16, 32x32, 48x48, 128x128 PNG icons & SVG without external dependencies.
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// CRC32 table
const crcTable = [];
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) {
    c = ((c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1));
  }
  crcTable[n] = c >>> 0;
}

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ crcTable[(crc ^ buf[i]) & 0xFF];
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function createChunk(type, data) {
  const len = data.length;
  const chunk = Buffer.alloc(12 + len);
  chunk.writeUInt32BE(len, 0);
  chunk.write(type, 4, 4, 'ascii');
  data.copy(chunk, 8);
  const typeAndData = chunk.subarray(4, 8 + len);
  const crc = crc32(typeAndData);
  chunk.writeUInt32BE(crc, 8 + len);
  return chunk;
}

function generateGoIconPNG(size) {
  const width = size;
  const height = size;
  // RGBA buffer with scanline filter byte per row
  const rowBytes = 1 + width * 4;
  const rawData = Buffer.alloc(height * rowBytes);

  const cx1 = size * 0.38;
  const cy1 = size * 0.44;
  const r1 = size * 0.32;

  const cx2 = size * 0.64;
  const cy2 = size * 0.58;
  const r2 = size * 0.32;

  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowBytes;
    rawData[rowOffset] = 0; // Filter type 0 (None)

    for (let x = 0; x < width; x++) {
      const pxOffset = rowOffset + 1 + x * 4;

      // Distance to Black stone center
      const d1 = Math.hypot(x - cx1, y - cy1);
      // Distance to White stone center
      const d2 = Math.hypot(x - cx2, y - cy2);

      let r = 0, g = 0, b = 0, a = 0;

      // Render White stone (in front)
      if (d2 <= r2) {
        const edgeFactor = Math.max(0, Math.min(1, (r2 - d2) * 1.5));
        const highlight = Math.max(0, 1 - (Math.hypot(x - (cx2 - r2 * 0.35), y - (cy2 - r2 * 0.35)) / r2));
        const val = Math.min(255, Math.round(210 + highlight * 45));
        r = val;
        g = val;
        b = val;
        a = Math.round(255 * edgeFactor);
      }
      // Render Black stone (behind)
      else if (d1 <= r1) {
        const edgeFactor = Math.max(0, Math.min(1, (r1 - d1) * 1.5));
        const highlight = Math.max(0, 1 - (Math.hypot(x - (cx1 - r1 * 0.35), y - (cy1 - r1 * 0.35)) / r1));
        const val = Math.round(25 + highlight * 65);
        r = val;
        g = val;
        b = val + 5;
        a = Math.round(255 * edgeFactor);
      }

      rawData[pxOffset] = r;
      rawData[pxOffset + 1] = g;
      rawData[pxOffset + 2] = b;
      rawData[pxOffset + 3] = a;
    }
  }

  // PNG Header
  const signature = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

  // IHDR
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData[8] = 8; // Bit depth
  ihdrData[9] = 6; // Color type 6 (RGBA)
  ihdrData[10] = 0; // Compression
  ihdrData[11] = 0; // Filter
  ihdrData[12] = 0; // Interlace
  const ihdrChunk = createChunk('IHDR', ihdrData);

  // IDAT
  const compressed = zlib.deflateSync(rawData);
  const idatChunk = createChunk('IDAT', compressed);

  // IEND
  const iendChunk = createChunk('IEND', Buffer.alloc(0));

  return Buffer.concat([signature, ihdrChunk, idatChunk, iendChunk]);
}

function generateSVG() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <defs>
    <radialGradient id="blackStoneGrad" cx="35%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#64748B"/>
      <stop offset="40%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#020617"/>
    </radialGradient>
    <radialGradient id="whiteStoneGrad" cx="35%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="70%" stop-color="#E2E8F0"/>
      <stop offset="100%" stop-color="#94A3B8"/>
    </radialGradient>
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="4" flood-opacity="0.5"/>
    </filter>
  </defs>
  <!-- Background Wood Aesthetic Circle -->
  <circle cx="64" cy="64" r="60" fill="#0F172A" stroke="#10B981" stroke-width="3"/>
  <!-- Black Stone -->
  <circle cx="48" cy="54" r="32" fill="url(#blackStoneGrad)" filter="url(#dropShadow)"/>
  <!-- White Stone -->
  <circle cx="78" cy="74" r="32" fill="url(#whiteStoneGrad)" filter="url(#dropShadow)"/>
</svg>`;
}

const iconsDir = path.join(__dirname);
[16, 32, 48, 128].forEach(size => {
  const png = generateGoIconPNG(size);
  fs.writeFileSync(path.join(iconsDir, `icon${size}.png`), png);
  console.log(`Generated icon${size}.png (${size}x${size})`);
});

fs.writeFileSync(path.join(iconsDir, 'icon.svg'), generateSVG());
console.log('Generated icon.svg');
