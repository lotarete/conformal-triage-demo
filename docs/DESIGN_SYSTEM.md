# Accountable Medical Triage — Design System
## Reference Document for Streamlit Build

---

## Design Philosophy

**"Maison Noire"** — A luxury editorial dark interface inspired by high-fashion brand aesthetics (Calvin Klein, Celine, Acne Studios). The dashboard should feel like a beautifully curated magazine spread, not a data science notebook. Every element earns its space. White space is generous. Typography does the heavy lifting.

**Core Principles:**
1. **Restraint over decoration** — No gradients, no glows, no noise textures. Let contrast and typography create hierarchy.
2. **White as the accent** — In a dark world, white is the loudest color. Use it sparingly for maximum impact.
3. **Burgundy as the signature** — A single warm hue that says "this isn't a template." Used for brand moments and critical data.
4. **Sharp containers, soft interactions** — Cards and sections are architectural (sharp edges). Buttons and inputs are touchable (slight softness).
5. **No sidebar. Ever.** — Full-width layout. The app breathes.

---

## Color Palette

### Backgrounds (layered depth system)
```
Page Background    #09090B   (zinc-950, near-black)
Surface / Cards    #141414   (elevated surface)
Surface Hover      #1C1C1C   (subtle lift on interaction)
Elevated Surface   #1F1F1F   (modals, popovers, active tabs)
```

### Borders & Dividers
```
Border Default     #262626   (zinc-800, subtle divider)
Border Emphasis    #3F3F46   (zinc-700, active states)
Border Muted       #1A1A1A   (barely visible structural lines)
```

### Text
```
Text Primary       #FAFAFA   (zinc-50, near-white — headings, key data)
Text Secondary     #A1A1AA   (zinc-400, body text, descriptions)
Text Muted         #71717A   (zinc-500, captions, labels, metadata)
```

### Accent Colors
```
Burgundy           #A62845   (signature brand color — emergency, critical moments)
Burgundy Muted     #421220   (burgundy at 20% — subtle card tints)
```

### Semantic Colors
```
Emerald / Trust    #50C878   (trustworthy verdict, model correct, CP coverage)
Amber / Warning    #FBBF24   (uncertain prediction sets, GP visit class, CP saves)
Orange             #FA7923   (urgent care class)
Burgundy           #A62845   (emergency class, miss states)
```

### Triage Class Colors
```
Self Care          #50C878   (emerald)
GP Visit           #FBBF24   (amber)
Urgent Care        #FA7923   (orange)
Emergency          #A62845   (burgundy)
```

---

## Typography

### Font: DM Sans (Google Fonts)

**Why DM Sans:** Low-contrast geometric sans-serif by Colophon Foundry. Compact without being cramped — built for information-dense layouts. Clean enough for Calvin Klein, readable enough for data.

### Type Scale
```
Section Heading    DM Sans  500 (Medium)    ##  markdown     #FAFAFA
Card Title         DM Sans  500 (Medium)    1.05rem          #FAFAFA
Body Text          DM Sans  400 (Regular)   0.92rem          #A1A1AA
Caption / Label    DM Sans  400 (Regular)   0.8rem uppercase #71717A
Stat Number        DM Sans  300 (Light)     2.5rem           #FAFAFA
```

### Key Typography Rules
- **NEVER use monospace.** DM Sans has tabular figures — use those.
- **Uppercase sparingly.** Only for section headings, category pills, captions.
- **Light weight (300) for impact.** Big numbers use light weight — confident enough to be thin.
- **No bold (700+) in body text.** Medium (500) is the heaviest for emphasis.

---

## Spacing & Layout

### Grid
```
Max content width     900px    (centered, generous margins)
Card padding          1.5rem
Card gap              1.25rem
Section gap           3rem
```

---

## Border Radius

### Mixed System: Sharp Containers + Soft Interactions
```
Cards / Containers     2px     (nearly square — architectural)
Buttons (CTA)          6px     (slight softness — touchable)
Input fields           6px     (matches buttons)
Pills / Tags           4px     (subtle rounding)
```

### Streamlit config.toml
```toml
[theme]
primaryColor = "#FAFAFA"
backgroundColor = "#09090B"
secondaryBackgroundColor = "#141414"
textColor = "#A1A1AA"
font = "sans serif"
baseRadius = "0.125rem"
```
