/**
 * csv.js — CSV parsing (Google Ads Editor import) and export
 *
 * Exports:
 *   parseAdsEditorCSV(csvText)  → [{campaign, adGroup}]
 *   buildImportCSV(newAdGroups) → CSV string ready for Google Ads Editor
 *   downloadCSV(newAdGroups)    → triggers browser file download
 */

/**
 * Parse a Google Ads Editor CSV export and extract Ad Group rows.
 * Handles quoted fields and flexible column ordering.
 */
function parseAdsEditorCSV(csvText) {
  const lines = csvText.split('\n').filter(l => l.trim());
  if (lines.length < 2) throw new Error('CSV appears empty or has no data rows.');

  const headers    = parseCSVLine(lines[0]).map(h => h.toLowerCase().trim());
  const campaignIdx = headers.findIndex(h => h.includes('campaign'));
  const adGroupIdx  = headers.findIndex(h =>
    h.includes('ad group') || h.includes('adgroup') || h === 'ad group name' || h === 'name'
  );

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols     = parseCSVLine(lines[i]);
    const campaign = campaignIdx >= 0 ? (cols[campaignIdx] || '').trim() : '';
    const adGroup  = adGroupIdx  >= 0 ? (cols[adGroupIdx]  || '').trim() : (cols[0] || '').trim();
    if (adGroup) rows.push({ campaign, adGroup });
  }

  // Deduplicate by adGroup name
  return [...new Map(rows.map(r => [r.adGroup, r])).values()];
}

/**
 * Parse a single CSV line, respecting quoted fields.
 */
function parseCSVLine(line) {
  const result = [];
  let current  = '';
  let inQuotes = false;

  for (const char of line) {
    if (char === '"')              { inQuotes = !inQuotes; continue; }
    if (char === ',' && !inQuotes) { result.push(current); current = ''; continue; }
    current += char;
  }
  result.push(current);
  return result;
}

/**
 * Build a Google Ads Editor-compatible CSV string from the new Ad Groups array.
 * Format: Campaign, Ad Group, Max CPC, Status (+ one row per keyword).
 */
function buildImportCSV(newAdGroups) {
  const rows = [['Campaign', 'Ad Group', 'Max CPC', 'Status', 'Keyword', 'Match Type']];

  for (const ag of newAdGroups) {
    // Ad Group row
    rows.push([ag.campaign, ag.adGroup, ag.maxCpc, 'Enabled', '', '']);
    // Keyword rows
    for (const kw of (ag.keywords || [])) {
      rows.push([ag.campaign, ag.adGroup, ag.maxCpc, 'Enabled', kw, 'Broad']);
    }
  }

  return rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
}

/**
 * Trigger a browser download of the import CSV.
 */
function downloadCSV(newAdGroups) {
  if (!newAdGroups || newAdGroups.length === 0) {
    alert('No new Ad Groups to export.');
    return;
  }

  const csv      = buildImportCSV(newAdGroups);
  const blob     = new Blob([csv], { type: 'text/csv' });
  const a        = document.createElement('a');
  a.href         = URL.createObjectURL(blob);
  a.download     = `new-ad-groups-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}