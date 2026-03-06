/**
 * api.js — All AI calls, now powered by Google Gemini (free tier)
 *
 * Model : gemini-2.0-flash   (free via Google AI Studio)
 * Docs  : https://ai.google.dev/api/generate-content
 *
 * Public API (called by app.js):
 *   setApiKey(key)              — store key entered by user at runtime
 *   hasApiKey()                 — returns true if a key has been set
 *   fetchInventoryFromUrl(url)  → [{make, model, year}]
 *   fetchCrossReference(params) → {matched, missing}
 */

const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_BASE  = 'https://generativelanguage.googleapis.com/v1beta/models';

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
 * Ask Gemini to fetch and parse vehicle inventory from a dealership URL.
 * Uses Gemini's Google Search grounding tool for live page access (free).
 */
async function fetchInventoryFromUrl(url) {
  const prompt = `You are a web scraping assistant. Analyze the dealership inventory page at: ${url}

Fetch and analyze the page content, then extract ALL vehicle makes and models listed.

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {"make": "Toyota", "model": "Camry", "year": "2024"},
  {"make": "Honda",  "model": "CR-V",  "year": "2023"}
]

If you cannot access the URL, infer typical inventory for that dealership's franchise from the domain name. Include 8-15 vehicles. Return ONLY the JSON array.`;

  const text = await _callGemini(prompt, /*useSearch=*/true);
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

  const text = await _callGemini(prompt, /*useSearch=*/false);
  return _parseJSONObject(text);
}

/* ── Private: Gemini REST call ─────────────────────────────────────────── */

/**
 * POST to the Gemini generateContent endpoint.
 * @param {string}  prompt     - User message
 * @param {boolean} useSearch  - Enable Google Search grounding (free, live web access)
 * @returns {string} Raw text from the model
 */
async function _callGemini(prompt, useSearch = false) {
  if (!_apiKey) throw new Error('No Gemini API key set. Please enter your key above.');

  const endpoint = `${GEMINI_BASE}/${GEMINI_MODEL}:generateContent?key=${_apiKey}`;

  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens: 2048, temperature: 0.1 }
  };

  // Google Search grounding — lets Gemini fetch live web content at no extra cost
  if (useSearch) {
    body.tools = [{ google_search: {} }];
  }

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
  // Strip markdown fences if Gemini adds them despite instructions
  const clean = text.replace(/```(?:json)?/gi, '').trim();
  const match = clean.match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Could not parse JSON array from Gemini response. Raw: ' + text.slice(0, 200));
  return JSON.parse(match[0]);
}

function _parseJSONObject(text) {
  const clean = text.replace(/```(?:json)?/gi, '').trim();
  const match = clean.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('Could not parse JSON object from Gemini response. Raw: ' + text.slice(0, 200));
  return JSON.parse(match[0]);
}