# AdSync — Dealership Inventory × Google Ads

AI-powered tool to cross-reference a car dealership's live inventory against
Google Ads Ad Groups and auto-generate new ones for any missing models.

Powered by **Google Gemini** (free tier via Google AI Studio).

## Project Structure

```
adsync/
├── index.html        ← Entry point (markup only)
├── css/
│   └── styles.css    ← All styles and CSS variables
└── js/
    ├── api.js        ← Gemini API calls (inventory scrape + cross-reference)
    ├── csv.js        ← Google Ads Editor CSV parsing and export
    ├── ui.js         ← DOM rendering, logging, step management, drag-and-drop
    └── app.js        ← Main orchestration (wires all modules together)
```

## Deploy to GitHub Pages

1. Create a new GitHub repository (public or private).
2. Push this folder's contents to the `main` branch:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/adsync.git
   git push -u origin main
   ```
3. Go to **Settings → Pages** in your repo.
4. Set **Source** to `Deploy from a branch`, branch `main`, folder `/ (root)`.
5. Click **Save**. Your app will be live at:
   `https://YOUR_USERNAME.github.io/adsync/`

## How to Use

1. **Step 0** — Enter your free Gemini API key (get one at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
   The key is stored in memory only and never persisted.

2. **Step 1** — Paste the dealership's inventory URL and click *Scan Inventory*.
   Gemini uses Google Search grounding to fetch and extract all vehicle models live.

3. **Step 2** — Export Ad Groups from Google Ads Editor:
   `File → Export → Spreadsheet (CSV)` then drag the file into the drop zone.

4. **Step 3** — Enter your campaign name, adjust the Ad Group name template and
   Max CPC if needed, then click *Cross-Reference & Generate*.

5. Review the results and click **⬇ Download Import CSV** to get a
   Google Ads Editor-compatible file ready to import directly.

## Module Responsibilities

| File | Responsibility |
|------|----------------|
| `api.js` | All Gemini API calls. Swap model or prompts here. |
| `csv.js` | CSV parsing logic and export format. Adjust columns here for different Ads Editor versions. |
| `ui.js`  | Every DOM update lives here. Change rendering without touching business logic. |
| `app.js` | Orchestrates the four-step flow and owns shared state. |

## Notes

- No server required — runs entirely in the browser.
- Uses `gemini-2.0-flash` which is free under Google AI Studio's free tier limits.
- The Gemini API key is entered at runtime in the UI and is never committed to your repo.
- Google Search grounding (used for live inventory scraping) is included in the free tier.