/**
 * api.js — All AI calls, powered by Google Gemini (free tier)
 *
 * Model : gemini-2.0-flash  (free via Google AI Studio)
 * Docs  : https://ai.google.dev/api/generate-content
 *
 * How inventory scraping works (no grounding tool required):
 *   1. Browser fetches the dealership page via allorigins.win (free CORS proxy)
 *   2. Raw HTML is stripped down to plain text
 *   3. Plain text is sent to Gemini to extract vehicle makes/models
 *   This keeps everything on the free tier.
 *
 * Public API (called by app.js):
 *   setApiKey(key)              — store key entered by user at runtime
 *   hasApiKey()                 — returns true if a key has been set
 *   fetchInventoryFromUrl(url)  → [{make, model, year}]
 *   fetchCrossReference(params) → {matched, missing}
 */

const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_BASE  = 'https://generativelanguage.googleapis.com/v1beta/models';
const CORS_PROXY   = 'https://api.allorigins.win/get?url=';

let _apiKey = '';

/** Called by app.js when the user saves their key in the UI. */
function setApiKey(key) {
  _apiKey = key.trim();
}

/** Returns true if a key has been set. */
function hasApiKey() {
  return _apiKey.length > 0;
}

/* ── Public: inventory scrape ──────────────────────────────────────────── */

/**
 * Fetch the dealership page via CORS proxy, strip to plain text,
 * then ask Gemini to extract all vehicle makes/models/years.
 */
async function fetchInventoryFromUrl(url) {
  // Step 1: fetch page HTML through CORS proxy
  const pageText = await _fetchPageText(url);

  // Step 2: send plain text to Gemini for extraction
  const prompt = `You are a vehicle inventory extraction assistant. Below is the plain text content scraped from a car dealership's inventory page.

Extract every unique vehicle listed. For each vehicle return its make, model, and year.

Page content:
---
${pageText.slice(0, 12000)}
---

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {"make": "Toyota", "model": "Camry", "year": "2024"},
  {"make": "Honda",  "model": "CR-V",  "year": "2023"}
]

Deduplicate by make+model (keep the most recent year if duplicates exist). Return ONLY the JSON array.`;

  const text = await _callGemini(prompt);
  return _parseJSONArray(text);
}

/* ── Public: cross-reference ───────────────────────────────────────────── */

/**
 * Ask Gemini to compare inventory vs. existing Ad Groups and identify gaps.
 */
async function fetchCrossReference({ inventoryModels, existingAdGroups, campaign, template, maxCpc }) {
  const inventoryList = inventoryModels.map(v => `${v.year} ${v.make} ${v.model}`).join('\n');
  const adGroupList   = existingAdGroups.map(ag => ag.adGroup).join('\n');

  const prompt = `You are a Google Ads expert. Cross-reference this dealership inventory with existing Ad Groups and identify what is missing.

INVENTORY (from dealership website):
${inventoryList}

EXISTING AD GROUPS (from Google Ads):
${adGroupList}

CAMPAIGN: ${campaign}
AD GROUP TEMPLATE: ${template}
MAX CPC: $${maxCpc}

For each inventory vehicle, check if an existing Ad Group covers that make+model (year does not need to match).

Return ONLY this JSON — no markdown fences, no explanation:
{
  "matched": [
    {"make":"Toyota","model":"Camry","existingAdGroup":"Toyota Camry - New Cars"}
  ],
  "missing": [
    {
      "make":"Honda","model":"CR-V","year":"2023",
      "newAdGroupName":"Honda CR-V - New Cars",
      "keywords":["honda cr-v dealer","new honda cr-v","honda cr-v for sale","buy honda cr-v","honda cr-v near me"]
    }
  ]
}

For each missing vehicle, apply the template (replace [Make] and [Model]) and generate 5 relevant search keywords. Return ONLY the JSON.`;

  const text = await _callGemini(prompt);
  return _parseJSONObject(text);
}

/* ── Private: CORS proxy page fetch ───────────────────────────────────── */

/**
 * Fetch a URL via allorigins.win and return stripped plain text.
 * allorigins.win is a free, open CORS proxy that works from any browser.
 */
async function _fetchPageText(url) {
  const proxyUrl = CORS_PROXY + encodeURIComponent(url);

  const res = await fetch(proxyUrl);
  if (!res.ok) throw new Error(`Could not fetch page (${res.status}). Check the URL and try again.`);

  const data = await res.json();
  const html = data?.contents;
  if (!html) throw new Error('CORS proxy returned empty content. The site may be blocking external access.');

  return _stripHtml(html);
}

/**
 * Strip HTML tags, scripts, styles, and collapse whitespace to plain text.
 * Keeps enough content for Gemini to identify vehicle listings.
 */
function _stripHtml(html) {
  // Remove script and style blocks entirely
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')           // strip remaining tags
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&#\d+;/g, ' ')
    .replace(/\s{2,}/g, ' ')            // collapse whitespace
    .trim();

  return text;
}

/* ── Private: Gemini REST call ─────────────────────────────────────────── */

/**
 * POST a prompt to the Gemini generateContent endpoint.
 * No tools used — stays on free tier.
 */
async function _callGemini(prompt) {
  if (!_apiKey) throw new Error('No Gemini API key set. Please enter your key above.');

  const endpoint = `${GEMINI_BASE}/${GEMINI_MODEL}:generateContent?key=${_apiKey}`;

  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens: 2048, temperature: 0.1 }
  };

  const res = await fetch(endpoint, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body)
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err?.error?.message || res.statusText;
    throw new Error(`Gemini API error ${res.status}: ${msg}`);
  }

  const data = await res.json();
  return _extractText(data);
}

/* ── Private: response parsing ─────────────────────────────────────────── */

function _extractText(geminiResponse) {
  // Gemini shape: candidates[0].content.parts[].text
  const parts = geminiResponse?.candidates?.[0]?.content?.parts || [];
  return parts.map(p => p.text || '').join('');
}

function _parseJSONArray(text) {
  const clean = text.replace(/```(?:json)?/gi, '').trim();
  const match = clean.match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Could not parse vehicle list from Gemini. Raw: ' + text.slice(0, 200));
  return JSON.parse(match[0]);
}

function _parseJSONObject(text) {
  const clean = text.replace(/```(?:json)?/gi, '').trim();
  const match = clean.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('Could not parse analysis from Gemini. Raw: ' + text.slice(0, 200));
  return JSON.parse(match[0]);
}