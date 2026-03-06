/**
 * app.js — Main orchestration layer
 *
 * Coordinates api.js, csv.js, and ui.js.
 * All onclick handlers in index.html point here.
 *
 * Global functions (called from HTML):
 *   saveApiKey()
 *   scrapeInventory()
 *   handleFile(file)
 *   crossReference()
 *   downloadCSV()
 *   reset()
 */

/* ── Shared state ──────────────────────────────────────────────────────── */
let inventoryModels  = [];   // [{make, model, year}]
let existingAdGroups = [];   // [{campaign, adGroup}]
let newAdGroups      = [];   // ad groups to create, built after cross-reference

/* ── Step 0: Save Gemini API key ───────────────────────────────────────── */
function saveApiKey() {
  const input  = document.getElementById('apiKeyInput');
  const status = document.getElementById('keyStatus');
  const key    = input.value.trim();

  if (!key) { status.textContent = '⚠ Please enter a key.'; status.style.color = 'var(--warn)'; return; }
  if (!key.startsWith('AIza')) {
    status.textContent = '⚠ That doesn\'t look like a valid Google AI Studio key (should start with "AIza").';
    status.style.color = 'var(--warn)';
    return;
  }

  setApiKey(key);   // api.js
  status.textContent = '✓ Key saved for this session.';
  status.style.color = 'var(--accent)';

  // Mask the input for visual security
  input.value = key.slice(0, 6) + '••••••••••••••••••••••••••••••••••';
  input.disabled = true;

  activateStep(1);
}

/* ── Step 1: Scrape inventory ──────────────────────────────────────────── */
async function scrapeInventory() {
  const url = document.getElementById('siteUrl').value.trim();
  if (!url) { alert('Please enter a dealership inventory URL.'); return; }
  if (!hasApiKey()) { alert('Please save your Gemini API key first (Step 0).'); return; }

  const btn   = document.querySelector('#step1 .btn');
  const label = document.getElementById('scrapeLabel');
  btn.disabled    = true;
  label.innerHTML = '<span class="spinner"></span> Scanning...';
  log('scrapeLog', `Fetching page via CORS proxy: ${url}`);
  log('scrapeLog', 'Stripping HTML and sending to Gemini...');

  try {
    inventoryModels = await fetchInventoryFromUrl(url);
    log('scrapeLog', `✓ Found ${inventoryModels.length} vehicle models`, 'ok');
    inventoryModels.forEach(v => log('scrapeLog', `  → ${v.year} ${v.make} ${v.model}`));
    activateStep(2);
  } catch (err) {
    log('scrapeLog', `Error: ${err.message}`, 'err');
  } finally {
    btn.disabled      = false;
    label.textContent = 'Scan Inventory';
  }
}

/* ── Step 2: Handle CSV file upload ────────────────────────────────────── */
function handleFile(file) {
  if (!file) return;
  log('csvLog', `Reading file: ${file.name}`);

  const reader   = new FileReader();
  reader.onload  = e => _processCSVText(e.target.result);
  reader.onerror = ()  => log('csvLog', 'Failed to read file.', 'err');
  reader.readAsText(file);
}

function _processCSVText(csvText) {
  try {
    existingAdGroups = parseAdsEditorCSV(csvText);

    log('csvLog', `✓ Parsed ${existingAdGroups.length} unique Ad Groups`, 'ok');
    existingAdGroups.slice(0, 5).forEach(ag => log('csvLog', `  → ${ag.adGroup}`));
    if (existingAdGroups.length > 5) {
      log('csvLog', `  ... and ${existingAdGroups.length - 5} more`);
    }

    // Auto-populate campaign name if detectable
    const detectedCampaign = existingAdGroups.find(ag => ag.campaign)?.campaign || '';
    if (detectedCampaign) {
      document.getElementById('campaignName').value = detectedCampaign;
    }

    activateStep(3);
  } catch (err) {
    log('csvLog', `Error: ${err.message}`, 'err');
  }
}

/* ── Step 3: Cross-reference and generate ──────────────────────────────── */
async function crossReference() {
  const campaign = document.getElementById('campaignName').value.trim();
  const template = document.getElementById('adGroupTemplate').value.trim() || '[Make] [Model] - New Cars';
  const maxCpc   = document.getElementById('maxCpc').value.trim() || '2.00';

  if (!campaign)                    { alert('Please enter a campaign name.'); return; }
  if (inventoryModels.length  === 0) { alert('No inventory data — please complete Step 1.'); return; }
  if (existingAdGroups.length === 0) { alert('No Ad Group data — please complete Step 2.'); return; }

  const btn   = document.querySelector('#step3 .btn');
  const label = document.getElementById('analyzeLabel');
  btn.disabled    = true;
  label.innerHTML = '<span class="spinner"></span> Analyzing...';
  log('analysisLog', 'Sending to Gemini for cross-reference analysis...');

  try {
    const { matched = [], missing = [] } = await fetchCrossReference({
      inventoryModels,
      existingAdGroups,
      campaign,
      template,
      maxCpc
    });

    log('analysisLog', `✓ ${matched.length} models already covered`, 'ok');
    log('analysisLog', `✦ ${missing.length} models need new Ad Groups`, missing.length > 0 ? 'ok' : 'info');

    newAdGroups = missing.map(m => ({
      campaign,
      adGroup:  m.newAdGroupName,
      maxCpc,
      make:     m.make,
      model:    m.model,
      keywords: m.keywords || []
    }));

    renderResults(matched, missing, newAdGroups, inventoryModels.length);
  } catch (err) {
    log('analysisLog', `Error: ${err.message}`, 'err');
  } finally {
    btn.disabled      = false;
    label.textContent = '✦ Cross-Reference & Generate';
  }
}

/* ── Export ────────────────────────────────────────────────────────────── */
function downloadCSV() {
  if (newAdGroups.length === 0) { alert('No new Ad Groups to export.'); return; }

  const csv  = buildImportCSV(newAdGroups);
  const blob = new Blob([csv], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `new-ad-groups-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── Reset ─────────────────────────────────────────────────────────────── */
function reset() {
  inventoryModels  = [];
  existingAdGroups = [];
  newAdGroups      = [];

  ['siteUrl', 'campaignName'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });

  ['scrapeLog', 'csvLog', 'analysisLog'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.innerHTML = ''; el.style.display = 'none'; }
  });

  const panel = document.getElementById('resultsPanel');
  if (panel) panel.classList.remove('visible');

  document.querySelectorAll('.step').forEach(s => s.classList.remove('done'));

  // Return to step 1 if key is already set, otherwise step 0
  activateStep(hasApiKey() ? 1 : 0);
}

/* ── Boot ──────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initDropZone();

  // Allow Enter key to submit API key
  document.getElementById('apiKeyInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') saveApiKey();
  });
});