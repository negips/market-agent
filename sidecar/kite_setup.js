/**
 * kite_setup.js — Obtain a fresh Kite Connect access token via browser automation.
 *
 * Run once per trading day before starting the Julia session:
 *   node sidecar/kite_setup.js
 *
 * Required env vars (read from repo-root .env automatically):
 *   KITE_API_KEY       — Kite Connect app key
 *   KITE_API_SECRET    — Kite Connect app secret
 *   KITE_USER_ID       — Zerodha client ID (e.g. AB1234)
 *   KITE_PASSWORD      — Zerodha login password
 *   KITE_TOTP_SECRET   — Base32 TOTP secret from your authenticator app setup
 *
 * Output:
 *   sidecar/kite_session.json  (gitignored) — contains access_token, valid for today
 *
 * The running sidecar exposes the token at GET /kite/token.
 */

import crypto   from 'crypto';
import fs       from 'fs';
import path     from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Load .env from repo root ──────────────────────────────────────────────────

try {
  const envFile = fs.readFileSync(path.join(__dirname, '..', '.env'), 'utf8');
  for (const line of envFile.split('\n')) {
    const m = line.match(/^([^#\s][^=]*)=(.*)$/);
    if (m && !process.env[m[1].trim()]) {
      process.env[m[1].trim()] = m[2].trim();
    }
  }
} catch { /* .env is optional */ }

// ── Config ────────────────────────────────────────────────────────────────────

const {
  KITE_API_KEY:     API_KEY,
  KITE_API_SECRET:  API_SECRET,
  KITE_USER_ID:     USER_ID,
  KITE_PASSWORD:    PASSWORD,
  KITE_TOTP_SECRET: TOTP_SECRET,
} = process.env;

const missing = ['KITE_API_KEY','KITE_API_SECRET','KITE_USER_ID','KITE_PASSWORD','KITE_TOTP_SECRET']
  .filter(k => !process.env[k]);
if (missing.length) {
  console.error(`Missing required env vars: ${missing.join(', ')}`);
  console.error('Add them to your .env file in the repo root.');
  process.exit(1);
}

const LOGIN_URL    = `https://kite.zerodha.com/connect/login?v=3&api_key=${API_KEY}`;
const TOKEN_URL    = 'https://api.kite.trade/session/token';
const SESSION_FILE = path.join(__dirname, 'kite_session.json');

// ── TOTP (RFC 6238) — no external packages ────────────────────────────────────

function generateTOTP(base32Secret) {
  // Base32 decode
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  const clean    = base32Secret.toUpperCase().replace(/[^A-Z2-7]/g, '');
  let bits = '';
  for (const ch of clean) bits += alphabet.indexOf(ch).toString(2).padStart(5, '0');
  const key = Buffer.from(
    Array.from({ length: Math.floor(bits.length / 8) },
               (_, i) => parseInt(bits.slice(i * 8, i * 8 + 8), 2))
  );

  // Counter = floor(unix_seconds / 30)
  const counter = Math.floor(Date.now() / 30_000);
  const msg     = Buffer.allocUnsafe(8);
  msg.writeBigUInt64BE(BigInt(counter));

  // HMAC-SHA1 → dynamic truncation → 6-digit code
  const h      = crypto.createHmac('sha1', key).update(msg).digest();
  const offset = h[19] & 0x0f;
  const code   = (h.readUInt32BE(offset) & 0x7fff_ffff) % 1_000_000;
  return code.toString().padStart(6, '0');
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  // Import Playwright from the tijori-finance-mcp install (avoids a second Chromium download)
  const { chromium } = await import('./tijori-finance-mcp/node_modules/playwright/index.js');

  console.log('Launching Chromium for Kite login...');
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page    = await context.newPage();

  let requestToken = null;

  // Intercept any navigation that carries request_token (works regardless of redirect URL)
  page.on('framenavigated', frame => {
    if (frame !== page.mainFrame()) return;
    try {
      const params = new URL(frame.url()).searchParams;
      if (params.has('request_token') && params.get('status') === 'success') {
        requestToken = params.get('request_token');
      }
    } catch { /* non-URL frames */ }
  });

  // ── Step 1: User ID + Password ────────────────────────────────────────────

  await page.goto(LOGIN_URL);
  await page.waitForSelector('input[type="text"], input#userid', { timeout: 15_000 });

  const userField = page.locator('input#userid, input[type="text"]').first();
  const passField = page.locator('input#password, input[type="password"]').first();

  await userField.fill(USER_ID);
  await passField.fill(PASSWORD);
  await page.keyboard.press('Enter');

  // ── Step 2: TOTP ──────────────────────────────────────────────────────────

  // Wait for the TOTP input to appear (Zerodha calls it "External TOTP" or just shows a 6-digit input)
  await page.waitForSelector('input[type="number"], input[maxlength="6"], input#totp', { timeout: 15_000 });

  // Generate TOTP right before filling so it's maximally fresh
  const totp = generateTOTP(TOTP_SECRET);
  console.log(`Generated TOTP: ${totp}  (${30 - (Math.floor(Date.now() / 1000) % 30)}s remaining)`);

  const totpField = page.locator('input[type="number"], input[maxlength="6"], input#totp').first();
  await totpField.fill(totp);
  await page.keyboard.press('Enter');

  // ── Step 3: Wait for OAuth redirect ───────────────────────────────────────

  console.log('Waiting for Kite OAuth redirect...');
  const deadline = Date.now() + 30_000;
  while (!requestToken && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 300));
  }

  await browser.close();

  if (!requestToken) {
    throw new Error('Did not receive request_token within 30s. Check credentials or TOTP secret.');
  }

  // ── Step 4: Exchange request_token → access_token ─────────────────────────

  console.log('Exchanging request token for access token...');
  const checksum = crypto.createHash('sha256')
    .update(API_KEY + requestToken + API_SECRET)
    .digest('hex');

  const resp = await fetch(TOKEN_URL, {
    method:  'POST',
    headers: { 'X-Kite-Version': '3', 'Content-Type': 'application/x-www-form-urlencoded' },
    body:    new URLSearchParams({ api_key: API_KEY, request_token: requestToken, checksum }).toString(),
  });

  const body = await resp.json();
  if (body.status !== 'success') {
    throw new Error(`Token exchange failed: ${body.message ?? JSON.stringify(body)}`);
  }

  const { access_token, user_id, user_name } = body.data;
  const today = new Date().toISOString().slice(0, 10);

  const session = { access_token, user_id, user_name, api_key: API_KEY, date: today };
  fs.writeFileSync(SESSION_FILE, JSON.stringify(session, null, 2));

  console.log(`\nKite session saved for ${user_name} (${user_id})`);
  console.log(`Token valid for: ${today}`);
  console.log(`Saved to: ${SESSION_FILE}`);
}

main().catch(err => {
  console.error('\nError:', err.message);
  process.exit(1);
});
