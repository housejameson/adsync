/**
 * ui.js — DOM rendering, logging, step management, and drag-and-drop
 *
 * Exports:
 *   log(logId, message, type)
 *   activateStep(n)
 *   renderResults(matched, missing, newAdGroups, totalInventory)
 *   initDropZone()
 */

/* ── Logging ───────────────────────────────────────────────────────────── */

/**
 * Append a timestamped line to a log panel.
 * @param {string} logId  - Element ID of the .log container
 * @param {string} msg    - Message to display
 * @param {'info'|'ok'|'err'} type
 */
function log(logId, msg, type = 'info') {
  const el = document.getElementById(logId);
  if (!el) return;
  el.style.display = 'block';

  const ts   = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.innerHTML = `<span class="ts">${ts}</span><span class="msg">${msg}</span>`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

/* ── Step management ───────────────────────────────────────────────────── */

/**
 * Mark all steps before n as done, activate step n.
 * Supports steps 0–3.
 */
function activateStep(n) {
  for (let i = 0; i <= 3; i++) {
    const el = document.getElementById('step' + i);
    if (!el) continue;
    el.classList.remove('active', 'done');
    if (i < n) el.classList.add('done');
  }
  const target = document.getElementById('step' + n);
  if (target) target.classList.add('active');
}

/* ── Results rendering ─────────────────────────────────────────────────── */

/**
 * Populate and reveal the results panel.
 */
function renderResults(matched, missing, newAdGroups, totalInventory) {
  const panel = document.getElementById('resultsPanel');
  if (!panel) return;
  panel.classList.add('visible');

  _renderStats(missing.length, matched.length, totalInventory);
  _renderNewModels(missing);
  _renderExistingModels(matched);
  _renderPreviewTable(newAdGroups);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function _renderStats(newCount, matchedCount, totalCount) {
  const el = document.getElementById('statsRow');
  if (!el) return;
  el.innerHTML = `
    <div class="stat">
      <div class="stat-val green">${newCount}</div>
      <div class="stat-label">New Ad Groups to Create</div>
    </div>
    <div class="stat">
      <div class="stat-val purple">${matchedCount}</div>
      <div class="stat-label">Already Covered</div>
    </div>
    <div class="stat">
      <div class="stat-val orange">${totalCount}</div>
      <div class="stat-label">Total Inventory Models</div>
    </div>
  `;
}

function _renderNewModels(missing) {
  const el = document.getElementById('newModelsGrid');
  if (!el) return;
  el.innerHTML = missing.length === 0
    ? '<div style="color:var(--muted);font-size:12px">All inventory models are covered ✓</div>'
    : missing.map(m => `
        <div class="model-card new-card">
          <div class="model-name">${_esc(m.make)} ${_esc(m.model)}</div>
          <span class="tag tag-new">NEW</span>
        </div>`).join('');
}

function _renderExistingModels(matched) {
  const el = document.getElementById('existingModelsGrid');
  if (!el) return;
  el.innerHTML = matched.length === 0
    ? '<div style="color:var(--muted);font-size:12px">None matched</div>'
    : matched.map(m => `
        <div class="model-card exists-card">
          <div class="model-name">${_esc(m.make)} ${_esc(m.model)}</div>
          <span class="tag tag-exists">✓</span>
        </div>`).join('');
}

function _renderPreviewTable(newAdGroups) {
  const tbody = document.getElementById('previewBody');
  if (!tbody) return;
  tbody.innerHTML = newAdGroups.length === 0
    ? '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:20px">No new Ad Groups needed</td></tr>'
    : newAdGroups.map(ag => `
        <tr>
          <td>${_esc(ag.campaign)}</td>
          <td>${_esc(ag.adGroup)}</td>
          <td>$${_esc(ag.maxCpc)}</td>
          <td style="color:var(--muted)">${(ag.keywords || []).slice(0, 3).map(_esc).join(', ')}</td>
        </tr>`).join('');
}

/* ── Drag & drop ───────────────────────────────────────────────────────── */

/**
 * Wire up drag-and-drop events on the #dropZone element.
 * Calls handleFile() (defined in app.js) when a file is dropped.
 */
function initDropZone() {
  const dz = document.getElementById('dropZone');
  if (!dz) return;

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragging'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('dragging'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);   // handleFile is defined in app.js
  });
}

/* ── Utility ───────────────────────────────────────────────────────────── */

/** Escape HTML to prevent injection in innerHTML templates. */
function _esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}