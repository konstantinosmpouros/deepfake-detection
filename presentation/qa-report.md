# Presentation QA Report

## Scope
- Files in scope: `presentation/index.html`, `presentation/app.js`, `presentation/styles.css`, `presentation/qa-report.md`
- Desktop QA targets: 1280x720 and 1440x900
- Local test server: `http://127.0.0.1:4173/index.html`
- Constraints honored: laptop-first only, no mobile pass, no scientific-content changes

## Matrix

| Area | Checked | Result | Notes |
|---|---|---:|---|
| Top nav | 6 nav buttons, brand, CTA | Pass | View switching, active state, footer index/name updates |
| Footer nav | Previous / Next | Pass | Disabled at the first/last section |
| RQ tabs | 4 tabs | Pass | Active tab state and content swap verified |
| Dataset controls | 2 buttons | Pass | Dataset swap and active state verified |
| Data topics | 4 tabs | Pass | Samples / Frequency / Embedding / Shortcut audit verified |
| Architecture selectors | 10 buttons | Pass | Each architecture selection updated the detail panel |
| Architecture stages | All stage buttons per architecture | Pass | Stage button sets rendered and stepped through per model |
| Architecture tabs | 3 tabs | Pass | Mechanism / Training / Evidence panel switching verified |
| Evaluation tabs | 4 tabs | Pass | ID / OOD / Robustness / XAI switching verified |
| Result tabs | 4 tabs | Pass | Result panels switched correctly |
| Compare dialog | Open, selects, close | Pass | Select changes propagated to comparison table |
| Image zoom | Open, close | Pass | Click-to-zoom and close button verified |
| Keyboard | ArrowLeft, ArrowRight, F | Pass with limit | Arrow navigation works; fullscreen unavailable in this browser runtime |
| Hash routing | Deep links | Pass | `#architectures/vit-lora` restored the correct view/state |
| Scrolling | View and panel scrolling | Pass | Internal scroll areas remain usable at both sizes |
| Disabled / focus / ARIA | Active, disabled, dialog state | Pass | Added visible state exposure for nav, dataset toggles, and dialogs |
| Assets | Presentation images | Pass | Image assets loaded with positive natural dimensions |

## Bugs Fixed

1. Active navigation and dataset toggle state were not exposed to assistive tech.
   - Fix: `presentation/app.js`
   - Changes: `aria-current` now updates on the active nav item; dataset buttons now reflect `aria-pressed`; zoomable images expose dialog semantics.

2. Compare and image dialog close controls used the same generic label.
   - Fix: `presentation/index.html`
   - Changes: distinct close labels/titles for comparison vs image zoom, plus dialog trigger metadata on the compare button.

3. Cached script version prevented the browser from picking up the updated JS during QA.
   - Fix: `presentation/index.html`
   - Changes: bumped the `app.js` cache key to force the updated runtime to load.

## Checks

- `node --check /Users/nikosgatos/Documents/Deepfake Detection/deepfake-detection/presentation/app.js` — passed
- `python3 /Users/nikosgatos/.codex/skills/ai-html-presentations/scripts/validate_deck.py presentation` — passed with 3 reviewed density warnings (`act-problem`, `act-verdict`, `act-next`); these are intentional interactive workbench screens rather than minimal linear slides
- Static local-asset scan — 34 referenced image assets, 0 missing

## Independent Regression

After the agent pass, the integrated version was served again on port 8766 and independently checked for:

- CTA → Data Lab navigation, URL hash, `aria-current`, and `aria-pressed`
- OOD dataset + Shortcut Audit combined state
- Patch architecture selection, stage selection, training tab, stable `scrollTop = 0`, and deep hash
- Footer transition from Architectures to Evaluation and robustness protocol state
- Results OOD tab, comparison dialog, model selection, `aria-expanded`, and generated comparison content
- Distinct comparison/image-dialog close controls and restored closed state
- `#architectures/vit-lora` deep-link restoration in the agent pass

All integrated regression checks passed. Browser console inspection reported no runtime errors.

## Residual Limits

- Fullscreen is not testable in this browser runtime because the Fullscreen API is unavailable here.
- No mobile QA was performed by request.
- No content, metric, or scientific claims were changed.
