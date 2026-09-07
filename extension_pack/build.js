/**
 * GoBot Extension Build & Packaging Script
 * Assembles a clean, validated distribution in dist/ and creates a ready-to-load ZIP file.
 * Zero external dependencies (uses Node.js built-ins: fs, path, zlib).
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT_DIR = __dirname;
const DIST_DIR = path.join(ROOT_DIR, 'dist');
const UNPACKED_DIR = path.join(DIST_DIR, 'unpacked');

// 1. Ensure Directories
if (fs.existsSync(DIST_DIR)) {
  fs.rmSync(DIST_DIR, { recursive: true, force: true });
}
fs.mkdirSync(DIST_DIR, { recursive: true });
fs.mkdirSync(UNPACKED_DIR, { recursive: true });
fs.mkdirSync(path.join(UNPACKED_DIR, 'icons'), { recursive: true });
fs.mkdirSync(path.join(UNPACKED_DIR, 'popup'), { recursive: true });

// 2. Validate Manifest
console.log('Validating manifest.json...');
const manifestContent = fs.readFileSync(path.join(ROOT_DIR, 'manifest.json'), 'utf8');
const manifest = JSON.parse(manifestContent);
if (manifest.manifest_version !== 3) {
  throw new Error(`Manifest version must be 3, found ${manifest.manifest_version}`);
}
console.log(`✓ Manifest V3 validated: ${manifest.name} v${manifest.version}`);

// 3. Files to Copy
const filesToCopy = [
  { src: 'manifest.json', dest: 'manifest.json' },
  { src: 'background.js', dest: 'background.js' },
  { src: 'content_script.js', dest: 'content_script.js' },
  { src: 'overlay.css', dest: 'overlay.css' },
  { src: 'engine_client.js', dest: 'engine_client.js' },
  { src: 'fast_mcts.js', dest: 'fast_mcts.js' },
  { src: 'popup/popup.html', dest: 'popup/popup.html' },
  { src: 'popup/popup.css', dest: 'popup/popup.css' },
  { src: 'popup/popup.js', dest: 'popup/popup.js' },
  { src: 'icons/icon16.png', dest: 'icons/icon16.png' },
  { src: 'icons/icon32.png', dest: 'icons/icon32.png' },
  { src: 'icons/icon48.png', dest: 'icons/icon48.png' },
  { src: 'icons/icon128.png', dest: 'icons/icon128.png' },
  { src: 'icons/icon.svg', dest: 'icons/icon.svg' }
];

console.log('Copying distribution assets to dist/unpacked/...');
filesToCopy.forEach(({ src, dest }) => {
  const srcPath = path.join(ROOT_DIR, src);
  const destPath = path.join(UNPACKED_DIR, dest);
  if (!fs.existsSync(srcPath)) {
    throw new Error(`Missing source file: ${srcPath}`);
  }
  fs.copyFileSync(srcPath, destPath);
  console.log(`  + ${dest}`);
});

// 4. Pure Node.js ZIP Generator (PKZip Specification)
function createZipArchive(sourceDir, zipFilePath) {
  console.log(`Packaging ZIP archive: ${path.relative(ROOT_DIR, zipFilePath)}...`);

  // CRC32 table
  const crcTable = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = ((c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1));
    }
    crcTable[n] = c >>> 0;
  }
  function calcCrc32(buf) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) {
      crc = (crc >>> 8) ^ crcTable[(crc ^ buf[i]) & 0xFF];
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function getFilesRecursively(dir, base = '') {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(file => {
      const filePath = path.join(dir, file);
      const relPath = base ? `${base}/${file}` : file;
      const stat = fs.statSync(filePath);
      if (stat && stat.isDirectory()) {
        results = results.concat(getFilesRecursively(filePath, relPath));
      } else {
        results.push({ fullPath: filePath, relPath: relPath.replace(/\\/g, '/') });
      }
    });
    return results;
  }

  const files = getFilesRecursively(sourceDir);
  const localFileChunks = [];
  const centralDirChunks = [];
  let currentOffset = 0;

  files.forEach(f => {
    const fileData = fs.readFileSync(f.fullPath);
    const uncompressedSize = fileData.length;
    const crc = calcCrc32(fileData);
    const compressedData = zlib.deflateRawSync(fileData);
    const compressedSize = compressedData.length;
    const nameBuf = Buffer.from(f.relPath, 'utf8');

    // Local File Header
    const localHeader = Buffer.alloc(30 + nameBuf.length);
    localHeader.writeUInt32LE(0x04034b50, 0); // Signature
    localHeader.writeUInt16LE(20, 4);         // Version needed
    localHeader.writeUInt16LE(0, 6);          // Flags
    localHeader.writeUInt16LE(8, 8);          // Compression method (Deflate)
    localHeader.writeUInt16LE(0, 10);         // Last mod time
    localHeader.writeUInt16LE(0, 12);         // Last mod date
    localHeader.writeUInt32LE(crc, 14);       // CRC32
    localHeader.writeUInt32LE(compressedSize, 18);   // Compressed size
    localHeader.writeUInt32LE(uncompressedSize, 22); // Uncompressed size
    localHeader.writeUInt16LE(nameBuf.length, 26);   // File name length
    localHeader.writeUInt16LE(0, 28);         // Extra field length
    nameBuf.copy(localHeader, 30);

    localFileChunks.push(localHeader, compressedData);

    // Central Directory Header
    const centralHeader = Buffer.alloc(46 + nameBuf.length);
    centralHeader.writeUInt32LE(0x02014b50, 0); // Signature
    centralHeader.writeUInt16LE(20, 4);          // Version made by
    centralHeader.writeUInt16LE(20, 6);          // Version needed
    centralHeader.writeUInt16LE(0, 8);           // Flags
    centralHeader.writeUInt16LE(8, 10);          // Compression method (Deflate)
    centralHeader.writeUInt16LE(0, 12);          // Last mod time
    centralHeader.writeUInt16LE(0, 14);          // Last mod date
    centralHeader.writeUInt32LE(crc, 16);        // CRC32
    centralHeader.writeUInt32LE(compressedSize, 20);
    centralHeader.writeUInt32LE(uncompressedSize, 24);
    centralHeader.writeUInt16LE(nameBuf.length, 28);
    centralHeader.writeUInt16LE(0, 30);          // Extra field length
    centralHeader.writeUInt16LE(0, 32);          // File comment length
    centralHeader.writeUInt16LE(0, 34);          // Disk number start
    centralHeader.writeUInt16LE(0, 36);          // Internal file attrs
    centralHeader.writeUInt32LE(0, 38);          // External file attrs
    centralHeader.writeUInt32LE(currentOffset, 42); // Relative offset of local header
    nameBuf.copy(centralHeader, 46);

    centralDirChunks.push(centralHeader);

    currentOffset += localHeader.length + compressedData.length;
  });

  const centralDirOffset = currentOffset;
  const centralDirSize = centralDirChunks.reduce((acc, c) => acc + c.length, 0);

  // End of Central Directory Record
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); // Signature
  eocd.writeUInt16LE(0, 4);          // Number of this disk
  eocd.writeUInt16LE(0, 6);          // Disk where central directory starts
  eocd.writeUInt16LE(files.length, 8); // Number of central directory records on this disk
  eocd.writeUInt16LE(files.length, 10); // Total number of central directory records
  eocd.writeUInt32LE(centralDirSize, 12); // Size of central directory
  eocd.writeUInt32LE(centralDirOffset, 16); // Offset of start of central directory
  eocd.writeUInt16LE(0, 20);         // Comment length

  const finalZipBuffer = Buffer.concat([...localFileChunks, ...centralDirChunks, eocd]);
  fs.writeFileSync(zipFilePath, finalZipBuffer);
  console.log(`✓ Zip package created successfully (${(finalZipBuffer.length / 1024).toFixed(1)} KB)`);
}

const zipPath = path.join(DIST_DIR, 'gobot-extension.zip');
createZipArchive(UNPACKED_DIR, zipPath);

console.log('\n======================================================');
console.log(' GoBot Extension Build Complete!');
console.log(` - Unpacked Folder : ${UNPACKED_DIR}`);
console.log(` - ZIP Distribution: ${zipPath}`);
console.log('======================================================\n');
