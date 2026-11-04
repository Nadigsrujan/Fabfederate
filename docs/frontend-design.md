# FabFederate React frontend design and integration

The user requested an animated research header, shadcn-compatible components, Tailwind, TypeScript and a clean frontend redesign. The later repository-organization request consolidates the complete React project under `frontend/`. The Python research workflows and saved model results remain authoritative.

## Compact design system

- Palette: blue ink #172A3D; white #FFFFFF; cool platinum #F3F6FA; steel #8796A7; cobalt #3567E8; quiet teal #237F84. Cyan/blue beams remain the single expressive element requested by the user.
- Typography: locally bundled IBM Plex Sans, 400/500/600, with tabular numeric results. Large 44–54px overview heading, 28px page titles, 18px section titles, 14–16px working text. Sentence-case labels.
- Layout: left-aligned workspace, persistent narrow navigation on desktop and compact scrollable navigation on mobile. One bounded animated header, then an asymmetric results chart and dataset record; other views prioritize charts, tables, and filters over repeated marketing cards.
- Behavior: real artifact selection, scenario/method filters, recorded model predictions, bounded SHAP, CSV upload, accessible loading/error/empty states. No fabricated live connectivity, model gains, or campus data.

```text
Desktop
+----------+--------------------------------------------------+
| Brand    | Workspace / selected experiment    Report download |
|          +--------------------------------------------------+
| Overview | Requested beams + project heading | validation plot|
| Results  +-----------------------------------+---------------+
| Clients  | Method comparison                 | Dataset record|
| Dataset  | measured test errors              | split coverage|
| Estimate +-----------------------------------+---------------+
|          | Latest experiment evidence / source limitations    |
+----------+--------------------------------------------------+

Mobile
+------------------------------------+
| Brand + experiment selector         |
| Overview Results Clients ...        |
+------------------------------------+
| Beams and concise project heading   |
| Method chart                        |
| Dataset record                      |
| Responsive tables / detail views    |
+------------------------------------+
```

## Review against the brief

A bounded animated header introduces the workspace without obscuring the research tasks. Actual validation traces, model comparisons and measured dataset coverage provide the subject-specific visuals. The canvas requires no image assets. Locally bundled IBM Plex Sans and Lucide supply typography and icons. The generic standalone demo was removed during organization; the reusable `ResearchBackground` renders application children.

## Engineering decisions

React/Vite/TypeScript are contained in `frontend/`, with `src/pages/` for application views, `src/components/ui/` for reusable components, `src/lib/utils.ts` for `cn`, and `src/styles/globals.css` for Tailwind and design tokens. The `@/` alias resolves to `frontend/src/` in Vite, TypeScript and Vitest. `frontend/components.json` declares these paths for shadcn. Python endpoints live under `src/fabfederate/api/`.

Canvas behavior: CSS-space coordinates at high DPR, container resizing, children rendering, animation-frame cleanup, reduced-motion support and hidden/offscreen pausing. No context provider or global store is required. App state is local plus the URL hash; API requests are cancelled when experiments change.

## Implementation checklist

- [x] Configure NPM, Vite, TypeScript, Tailwind and shadcn-compatible aliases.
- [x] Integrate the research header with sizing and reduced-motion checks.
- [x] Implement the local artifact API with path/input validation and tests.
- [x] Build overview, experiments, client, dataset and inference views using real artifacts.
- [x] Verify responsive browser flows, production build and regression tests.
- [x] Document setup and exact paths in the single root README.
