/**
 * Tijori Finance HTTP Sidecar
 *
 * Wraps the tijori-finance-mcp browser tools behind a plain HTTP/JSON API so
 * that the Julia trading agent can call them with HTTP.jl instead of speaking
 * the MCP protocol.
 *
 * Setup (one-time):
 *   git clone https://github.com/LaZZy0v0/tijori-finance-mcp.git
 *   cd tijori-finance-mcp && node setup.js   # authenticates and saves session
 *   cd ..
 *   npm install
 *
 * Run:
 *   node server_http.js          # default port 3001
 *   PORT=3002 node server_http.js
 *
 * All responses share the envelope:
 *   { ok: true,  data: <payload> }
 *   { ok: false, error: "<message>" }
 */

import express        from 'express';
import { readFileSync } from 'fs';
import path             from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Tijori tool imports ───────────────────────────────────────────────────────
// All paths are relative to THIS file. The tijori-finance-mcp clone must live
// at ./tijori-finance-mcp/ (sibling of this file). Imports resolve correctly
// because ES module relative paths are always relative to the importing file.

import { searchCompany }                      from './tijori-finance-mcp/src/tools/search.js';
import {
  getCompanyOverview,
  getKnowledgeBase,
  fetchDocument,
  getRevenueMix,
  getMarketShare,
}                                             from './tijori-finance-mcp/src/tools/company.js';
import { getFinancials }                      from './tijori-finance-mcp/src/tools/financials.js';
import { getShareholding }                    from './tijori-finance-mcp/src/tools/shareholding.js';
import {
  getOperationalMetrics,
  getFundFlow,
}                                             from './tijori-finance-mcp/src/tools/metrics.js';
import {
  listPopularScreens,
  screenCompanies,
  searchScreenerFields,
}                                             from './tijori-finance-mcp/src/tools/screener.js';
import {
  getMarkets,
  getNicheConstituents,
  getConglomerateConstituents,
  getRawMaterials,
  getMacroIndicators,
}                                             from './tijori-finance-mcp/src/tools/market.js';

// ── App setup ─────────────────────────────────────────────────────────────────

const app  = express();
const PORT = parseInt(process.env.PORT ?? '3001', 10);

app.use(express.json());

// ── Response helpers ──────────────────────────────────────────────────────────

/** Wrap an async tool call and return a consistent JSON envelope. */
function route(fn) {
  return async (req, res) => {
    try {
      const data = await fn(req);
      res.json({ ok: true, data });
    } catch (err) {
      console.error(`[ERROR] ${req.method} ${req.path} —`, err.message);
      res.status(500).json({ ok: false, error: err.message });
    }
  };
}

// ── Health ────────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'running', port: PORT } });
});

/**
 * POST /shutdown
 * Gracefully shut down the sidecar. Node.js handles Playwright/Chromium
 * cleanup before exiting, which kill() from the parent cannot guarantee.
 */
/**
 * GET /kite/token
 * Returns today's Kite Connect access token from kite_session.json.
 * Run `node kite_setup.js` once per trading day to refresh it.
 */
app.get('/kite/token', route(_req => {
  let session;
  try {
    session = JSON.parse(readFileSync(path.join(__dirname, 'kite_session.json'), 'utf8'));
  } catch {
    throw new Error('No Kite session found. Run: node sidecar/kite_setup.js');
  }
  const today = new Date().toISOString().slice(0, 10);
  if (session.date !== today) {
    throw new Error(`Kite session expired (was for ${session.date}). Run: node sidecar/kite_setup.js`);
  }
  return {
    access_token: session.access_token,
    user_id:      session.user_id,
    user_name:    session.user_name,
    valid_for:    session.date,
    acquired_at:  session.acquired_at,
  };
}));

app.post('/shutdown', (_req, res) => {
  res.json({ ok: true, data: { status: 'shutting down' } });
  setTimeout(() => process.exit(0), 100);
});

// ── Company search & overview ─────────────────────────────────────────────────

/**
 * GET /search?q=<name>
 * Search for companies by name. Returns slug, display name, and symbol.
 */
app.get('/search', route(req => {
  const q = req.query.q?.trim();
  if (!q) throw new Error('Missing required query param: q');
  return searchCompany(q);
}));

/**
 * GET /overview?slug=<company-slug>
 * Company overview: key ratios, forensics score, market cap, revenue mix.
 */
app.get('/overview', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getCompanyOverview(slug);
}));

/**
 * GET /resolve?slug=<company-slug>
 * Resolve a slug to its numeric company_id (needed for fund flow).
 * Extracts company_id from the overview page — same data, no extra load.
 */
app.get('/resolve', route(async req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  const overview = await getCompanyOverview(slug);
  const company_id = overview?.company_id ?? null;
  if (!company_id) throw new Error(`Could not resolve company_id for slug: ${slug}`);
  return { slug, company_id };
}));

// ── Financials ────────────────────────────────────────────────────────────────

/**
 * GET /financials?slug=<slug>&type=<pl|bs|cf|ratios|quarterly>
 *
 * type:
 *   pl        — Profit & Loss (annual)
 *   bs        — Balance Sheet (annual)
 *   cf        — Cash Flow Statement (annual)
 *   ratios    — Key financial ratios (annual)
 *   quarterly — Quarterly P&L results
 */
app.get('/financials', route(req => {
  const { slug, type } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  if (!type) throw new Error('Missing required query param: type (pl|bs|cf|ratios|quarterly)');
  return getFinancials(slug, type);
}));

// ── Shareholding ──────────────────────────────────────────────────────────────

/**
 * GET /shareholding?slug=<slug>
 * 10-quarter breakdown: Promoter, Promoter Pledged, FII, DII, Public.
 */
app.get('/shareholding', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getShareholding(slug);
}));

// ── Operational metrics & fund flow ──────────────────────────────────────────

/**
 * GET /metrics?slug=<slug>
 * All operational KPIs with full historical time series.
 */
app.get('/metrics', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getOperationalMetrics(slug);
}));

/**
 * GET /fundflow?company_id=<id>&years=<1|3|5|7|10>
 * Capital allocation breakdown over the specified horizon.
 */
app.get('/fundflow', route(req => {
  const company_id = parseInt(req.query.company_id, 10);
  const years      = parseInt(req.query.years ?? '5', 10);
  if (isNaN(company_id)) throw new Error('Missing or invalid query param: company_id (integer)');
  return getFundFlow(company_id, years);
}));

// ── Revenue mix & market share ────────────────────────────────────────────────

/**
 * GET /revenuemix?slug=<slug>
 * Segment revenue breakdown with historical trend per segment.
 */
app.get('/revenuemix', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getRevenueMix(slug);
}));

/**
 * GET /marketshare?slug=<slug>
 * Market share % per metric with as-of date.
 */
app.get('/marketshare', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getMarketShare(slug);
}));

// ── Knowledge base & document fetching ───────────────────────────────────────

/**
 * GET /knowledge?slug=<slug>
 * Annual reports, earnings releases, investor presentations, conference calls.
 * Returns authenticated CDN URLs grouped by document type.
 */
app.get('/knowledge', route(req => {
  const { slug } = req.query;
  if (!slug) throw new Error('Missing required query param: slug');
  return getKnowledgeBase(slug);
}));

/**
 * POST /document
 * Body: { "url": "https://files.tijorifinance.com/..." }
 * Fetches and extracts full text from an authenticated Tijori PDF.
 * Only files.tijorifinance.com URLs are accepted.
 */
app.post('/document', route(req => {
  const { url } = req.body;
  if (!url) throw new Error('Missing required body field: url');
  return fetchDocument(url);
}));

// ── Screener ──────────────────────────────────────────────────────────────────

/**
 * GET /screens
 * List all Tijori pre-built stock screens grouped by category.
 */
app.get('/screens', route(_req => listPopularScreens()));

/**
 * POST /screen
 * Body (one of):
 *   { "filters": "( ROCE > 20 ) and ( Market Capitalization > 1000 )" }
 *   { "preset": "Monopoly Companies" }
 *   { "alternate": "market share > 50" }
 *   { "filters": "...", "offset": 50, "limit": 50 }   (pagination)
 */
app.post('/screen', route(req => screenCompanies(req.body)));

/**
 * GET /fields?q=<query>[&type=<financials|...>]
 * Search Tijori's ~1,500 metric field catalog.
 */
app.get('/fields', route(req => {
  const q = req.query.q?.trim();
  if (!q) throw new Error('Missing required query param: q');
  return searchScreenerFields({ query: q, type: req.query.type });
}));

// ── Market data & macro ───────────────────────────────────────────────────────

/**
 * GET /markets?tab=<headline|niche|conglomerates>
 * Index performance data for the specified tab.
 *   headline      — Nifty 50, Bank Nifty, sectoral indices
 *   niche         — Tijori niche sector indices (includes tjiid for drill-down)
 *   conglomerates — business group indices (includes tjiid for drill-down)
 */
app.get('/markets', route(req => {
  const { tab } = req.query;
  if (!tab) throw new Error('Missing required query param: tab (headline|niche|conglomerates)');
  return getMarkets(tab);
}));

/**
 * GET /sector?tjiid=<id>
 * All stocks inside a Tijori niche sector index.
 * Get tjiid from GET /markets?tab=niche
 */
app.get('/sector', route(req => {
  const { tjiid } = req.query;
  if (!tjiid) throw new Error('Missing required query param: tjiid');
  return getNicheConstituents(tjiid);
}));

/**
 * GET /conglomerate?tjiid=<id>
 * All companies inside a business group (e.g. Tata, Reliance).
 * Get tjiid from GET /markets?tab=conglomerates
 */
app.get('/conglomerate', route(req => {
  const { tjiid } = req.query;
  if (!tjiid) throw new Error('Missing required query param: tjiid');
  return getConglomerateConstituents(tjiid);
}));

/**
 * GET /macro?tab=<industry|demand|gdp>
 * India macro indicators by category.
 *   industry — IIP, credit growth, GST collections
 *   demand   — auto sales, consumer data
 *   gdp      — GDP and trade data
 */
app.get('/macro', route(req => {
  const { tab } = req.query;
  if (!tab) throw new Error('Missing required query param: tab (industry|demand|gdp)');
  return getMacroIndicators(tab);
}));

/**
 * GET /rawmaterials?tab=<chemicals|spreads|metals>
 * Commodity price performance by category.
 */
app.get('/rawmaterials', route(req => {
  const { tab } = req.query;
  if (!tab) throw new Error('Missing required query param: tab (chemicals|spreads|metals)');
  return getRawMaterials(tab);
}));

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Tijori sidecar listening on http://localhost:${PORT}`);
  console.log('Endpoints:');
  console.log('  GET  /health');
  console.log('  GET  /kite/token');
  console.log('  GET  /search?q=');
  console.log('  GET  /overview?slug=');
  console.log('  GET  /financials?slug=&type=<pl|bs|cf|ratios|quarterly>');
  console.log('  GET  /shareholding?slug=');
  console.log('  GET  /metrics?slug=');
  console.log('  GET  /fundflow?company_id=&years=');
  console.log('  GET  /revenuemix?slug=');
  console.log('  GET  /marketshare?slug=');
  console.log('  GET  /knowledge?slug=');
  console.log('  POST /document  { url }');
  console.log('  GET  /screens');
  console.log('  POST /screen    { filters | preset | alternate }');
  console.log('  GET  /fields?q=');
  console.log('  GET  /markets?tab=<headline|niche|conglomerates>');
  console.log('  GET  /sector?tjiid=');
  console.log('  GET  /conglomerate?tjiid=');
  console.log('  GET  /macro?tab=<industry|demand|gdp>');
  console.log('  GET  /rawmaterials?tab=<chemicals|spreads|metals>');
});
