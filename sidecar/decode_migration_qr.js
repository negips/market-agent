/**
 * decode_migration_qr.js — Extract TOTP secrets from a Google Authenticator
 * migration QR code image.
 *
 * Usage:
 *   node sidecar/decode_migration_qr.js /path/to/migration_qr.png
 *
 * Requires: zbarimg (already installed — part of zbar package)
 *
 * Decodes the otpauth-migration://offline?data=<base64-protobuf> payload
 * entirely in-process using Node.js built-ins. No external npm packages needed.
 */

import { execSync }     from 'child_process';
import { readFileSync } from 'fs';

// ── Args ──────────────────────────────────────────────────────────────────────

const imgPath = process.argv[2];
if (!imgPath) {
  console.error('Usage: node sidecar/decode_migration_qr.js /path/to/qr.png');
  process.exit(1);
}

// ── Decode QR code → raw URL ──────────────────────────────────────────────────

let rawUrl;
try {
  rawUrl = execSync(`zbarimg --quiet --raw ${JSON.stringify(imgPath)}`, { encoding: 'utf8' }).trim();
} catch (err) {
  console.error('zbarimg failed:', err.message);
  process.exit(1);
}

if (!rawUrl.startsWith('otpauth-migration://')) {
  // Standard otpauth:// QR — just extract secret directly
  const params = new URL(rawUrl).searchParams;
  const secret = params.get('secret');
  const name   = decodeURIComponent(new URL(rawUrl).pathname.slice(1));
  console.log('\nStandard TOTP QR code detected:');
  console.log(`  Name:   ${name}`);
  console.log(`  Secret: ${secret}`);
  console.log(`\nAdd to .env:\n  KITE_TOTP_SECRET=${secret}`);
  process.exit(0);
}

// ── Decode protobuf payload ───────────────────────────────────────────────────
// Schema (Google Authenticator MigrationPayload):
//   repeated OtpParameters otp_parameters = 1;
//   message OtpParameters {
//     bytes  secret   = 1;
//     string name     = 2;
//     string issuer   = 3;
//     varint algorithm = 4;
//     varint digits    = 5;
//     varint type      = 6;
//   }

function readVarint(buf, pos) {
  let result = 0n, shift = 0n;
  while (pos < buf.length) {
    const byte = buf[pos++];
    result |= BigInt(byte & 0x7f) << shift;
    shift += 7n;
    if (!(byte & 0x80)) break;
  }
  return { value: result, pos };
}

function readLenDelim(buf, pos) {
  const { value: len, pos: p } = readVarint(buf, pos);
  const end = p + Number(len);
  return { value: buf.slice(p, end), pos: end };
}

function parseProto(buf) {
  const fields = {};
  let pos = 0;
  while (pos < buf.length) {
    const { value: tag, pos: p1 } = readVarint(buf, pos);
    pos = p1;
    const num      = Number(tag >> 3n);
    const wireType = Number(tag & 7n);
    if (wireType === 0) {
      const { value, pos: p2 } = readVarint(buf, pos);
      pos = p2;
      (fields[num] ??= []).push(value);
    } else if (wireType === 2) {
      const { value, pos: p2 } = readLenDelim(buf, pos);
      pos = p2;
      (fields[num] ??= []).push(value);
    } else {
      // Unknown wire type — stop parsing this message
      break;
    }
  }
  return fields;
}

function toBase32(bytes) {
  const alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  let bits = '';
  for (const b of bytes) bits += b.toString(2).padStart(8, '0');
  let out = '';
  for (let i = 0; i + 5 <= bits.length; i += 5)
    out += alpha[parseInt(bits.slice(i, i + 5), 2)];
  return out;
}

// ── Parse and display ─────────────────────────────────────────────────────────

const b64  = new URL(rawUrl).searchParams.get('data');
const buf  = Buffer.from(b64, 'base64');
const top  = parseProto(buf);
const entries = (top[1] ?? []).map(msgBuf => {
  const f      = parseProto(msgBuf);
  const secret = f[1]?.[0] ? toBase32(f[1][0]) : '(unknown)';
  const name   = f[2]?.[0]?.toString('utf8') ?? '';
  const issuer = f[3]?.[0]?.toString('utf8') ?? '';
  return { name, issuer, secret };
});

if (entries.length === 0) {
  console.error('No OTP entries found in migration QR.');
  process.exit(1);
}

console.log(`\nFound ${entries.length} account(s) in migration QR:\n`);
entries.forEach((e, i) => {
  console.log(`  ${i + 1}. ${e.issuer || e.name}`);
  if (e.issuer && e.name !== e.issuer) console.log(`       Account: ${e.name}`);
  console.log(`       Secret:  ${e.secret}`);
});

console.log('\nTo use Zerodha/Kite, find the matching entry above and add to .env:');
console.log('  KITE_TOTP_SECRET=<secret from above>');
