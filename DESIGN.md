---
name: Roadwatch
description: Evidence-first driver safety review presented as a disciplined vehicle inspection record.
colors:
  paper: "#ffffff"
  field: "#f4f4f4"
  ink: "#161616"
  muted: "#525252"
  rule: "#c6c6c6"
  strong-rule: "#8d8d8d"
  safety-green: "#198038"
  safety-green-dark: "#0e6027"
  warning-amber: "#b28600"
  failure-red: "#da1e28"
  rail-graphite: "#18201e"
  rail-hover: "#26302d"
  rail-active: "#286443"
  focus-blue: "#0f62fe"
typography:
  display:
    fontFamily: "IBM Plex Sans Condensed, sans-serif"
    fontSize: "clamp(2.5rem, 5vw, 4.7rem)"
    fontWeight: 600
    lineHeight: 0.95
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "IBM Plex Sans Condensed, sans-serif"
    fontSize: "clamp(2rem, 4vw, 3.35rem)"
    fontWeight: 600
    lineHeight: 0.95
    letterSpacing: "-0.025em"
  title:
    fontFamily: "IBM Plex Sans Condensed, sans-serif"
    fontSize: "21px"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "0.01em"
  body:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: "normal"
    letterSpacing: "normal"
  label:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: "normal"
    letterSpacing: "0.04em"
  metric:
    fontFamily: "IBM Plex Sans Condensed, sans-serif"
    fontSize: "37px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "normal"
rounded:
  compact: "3px"
  record: "6px"
  pill: "16px"
spacing:
  xxs: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  record: "18px"
  lg: "24px"
  shell-inline: "34px"
  shell-bottom: "48px"
components:
  button-primary:
    backgroundColor: "{colors.safety-green}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.record}"
    padding: "0 64px 0 16px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.safety-green-dark}"
    textColor: "{colors.paper}"
    rounded: "{rounded.record}"
  input-field:
    backgroundColor: "{colors.field}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "6px 6px 0 0"
    padding: "0 16px"
    height: "40px"
  record-panel:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.record}"
    padding: "{spacing.record}"
  nav-item-active:
    backgroundColor: "{colors.rail-active}"
    textColor: "{colors.paper}"
    rounded: "{rounded.record}"
    padding: "0 13px"
    height: "46px"
  metric-block:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.metric}"
    padding: "{spacing.record}"
    height: "106px"
  status-tag-confirmed:
    backgroundColor: "#a7f0ba"
    textColor: "{colors.safety-green-dark}"
    rounded: "{rounded.pill}"
    padding: "0 8px"
    height: "24px"
  notification-warning:
    backgroundColor: "#fcf4d6"
    textColor: "{colors.ink}"
    rounded: "{rounded.record}"
    padding: "14px 16px"
---

# Design System: Roadwatch

## Overview

**Creative North Star: "The Inspection Docket"**

Roadwatch presents machine-assisted safety review as a sober vehicle inspection record. Cool paper, graphite rules, a dark instrument rail, condensed operational headings, and compact tabular records make readiness and evidence feel inspectable rather than theatrical. The interface deliberately avoids the visual language of a CCTV spectacle or neon AI command center.

The system is dense without becoming cramped. Strong page headings establish the current task, while repeated bordered sheets, definition lists, and ledger rows keep facts aligned and comparable. Color is semantic and scarce; structure comes from typography, one-pixel rules, and tonal separation rather than decoration or elevation.

**Key Characteristics:**

- Operate-mode density built for scanning, comparison, and defensible review.
- Condensed uppercase headings paired with a quiet workhorse sans for records and controls.
- Flat white sheets on a cool-gray field, separated by crisp graphite rules.
- Safety green reserved for available, confirmed, or primary-action states.
- Explicit loading, empty, warning, error, and review states with no fabricated evidence.

## Colors

The palette is a controlled inspection-room neutral system with one operational green and tightly bounded warning and failure colors.

### Primary

- **Safety Green:** Marks the primary action, active navigation, operational icons, distribution bars, and confirmed or available states.
- **Deep Safety Green:** Deepens the primary action on hover and supplies readable green text links and confirmed-state text.

### Secondary

- **Inspection Amber:** Signals attention, ambiguous evidence, and no-seatbelt markers without implying system failure.

### Tertiary

- **Failure Red:** Is reserved for errors, failed operations, and explicit rejection treatments.
- **Focus Blue:** Provides the universal two-pixel keyboard focus indicator and remains distinct from operational status colors.

### Neutral

- **Cool Field:** The application canvas and table-header tone.
- **White Record Paper:** The panel, sheet, and form surface.
- **Graphite Ink:** Primary text and square evidence markers.
- **Muted Graphite:** Supporting copy, metadata, placeholders, and secondary labels.
- **Cool Rule / Strong Rule:** Standard panel dividers and stronger page-level boundaries.
- **Rail Graphite:** The persistent desktop rail, mobile header, and login principles field.
- **Rail Hover / Rail Active:** Local navigation feedback within the graphite rail.

### Named Rules

**The Evidence Color Rule.** Green means available or confirmed, amber means attention, and red means failure; every status also carries readable text, an icon, or both.

**The Cool Paper Rule.** Work surfaces stay white or cool gray. Saturated color never becomes a decorative panel fill.

## Typography

**Display Font:** IBM Plex Sans Condensed (with sans-serif fallback)  
**Body Font:** IBM Plex Sans (with system-ui and sans-serif fallbacks)

**Character:** The condensed face gives headings the clipped authority of an inspection form, while the regular sans keeps dense evidence and control text neutral. The pairing is direct, technical, and deliberately free of futurist styling.

### Hierarchy

- **Display:** Semibold, tightly tracked, and uppercase; used for the login statement and other rare, oversized operational declarations.
- **Headline:** Semibold, tightly tracked, and uppercase; used once per application page.
- **Title:** Semibold condensed uppercase; used for panel headings and record-sheet titles.
- **Body:** Regular workhorse sans; used for descriptions, values, control content, and explanatory copy, with long descriptions constrained to roughly 65 characters.
- **Label:** Compact uppercase sans with expanded tracking; used for table headers and definition terms.
- **Metric:** Medium-weight condensed numerals with a unit line-height; used only for operational counts.

### Named Rules

**The Operational Heading Rule.** Uppercase belongs to page titles, panel titles, table headers, and definition terms; sentences, helper text, and decisions remain in normal case.

**The Tabular Evidence Rule.** Counts, timestamps, identifiers, definition values, and strong values use tabular figures so columns do not shift while data changes.

## Layout

Desktop screens use a fixed 232px left rail and a centered work area capped at 1480px. The main canvas has 34px inline padding, 30px top padding, and 48px bottom padding. Page headers separate title and action with a strong horizontal rule; primary actions sit at the upper right and stay on one line.

The overview working field uses a 12px gap and a two-column ratio of 1.7fr to 0.9fr. The metric strip is four equal facts at desktop, two columns below 1050px, and one column below 760px. Upload, inspection, and other split workspaces collapse to one column below 1050px. Tables retain their density and scroll horizontally instead of crushing columns.

At 760px and below, the rail becomes a slide-in drawer behind a 46% black backdrop, the shell loses its left offset, and a sticky 58px graphite header exposes the menu control. Main padding becomes 22px by 16px by 40px, page-header actions expand to full width, facts stack, filter controls stack, and frame metadata becomes a vertical list. Drawer focus moves to the first navigation item and Escape returns focus to the trigger.

**The Twelve-Pixel Working Gap Rule.** Dashboard sheets and split workspaces use the same compact 12px seam so the screen reads as one record, not a collection of floating cards.

## Elevation & Depth

Roadwatch is flat by design. Panels, tables, metric strips, and forms use tonal layering and one-pixel rules; there are no resting card shadows. The only overlay depth is the mobile navigation backdrop. Keyboard focus uses a blue outline or Carbon's inset focus treatment, never a decorative shadow. Loading depth is expressed with restrained gray shimmer bands, and the shimmer is suppressed when reduced motion is requested.

### Named Rules

**The Flat Record Rule.** A surface earns separation through paper tone, border, or divider; do not add drop shadows to routine application sheets.

## Shapes

The system uses gently compact 6px corners on buttons, panels, notifications, navigation items, file selections, and evidence containers. Inputs use the same radius only on their top corners and retain a straight bottom rule. Status tags are the sole pill form, while event-type markers remain small outlined squares. Skeleton lines use an even tighter 3px corner.

**The One Radius Rule.** Six pixels is the default silhouette for interactive controls and record sheets; larger, softer card radii do not belong in this world.

## Components

### Buttons

- **Shape:** Compact six-pixel corners with Carbon's 48px default action height and asymmetric space reserved for an optional trailing icon.
- **Primary:** Safety-green fill, white label, and a deep-green hover state; used for upload, analysis, sign-in, review, and confirmation actions.
- **Hover / Focus:** Hover changes fill without lifting the control. Keyboard focus uses the visible blue system focus treatment.
- **Secondary / Tertiary / Ghost:** Carbon secondary, tertiary, and ghost treatments carry supporting decisions, exports, recovery, and quiet panel actions. Danger-tertiary is restricted to rejection.

### Chips

- **Style:** Carbon status tags are compact 24px pills with explicit status text. Confirmed uses a pale green field and deep-green text; pending, needs-review, rejected, and pre-training use their mapped gray, warm-gray, or red variants.
- **State:** Tags report state; they are not used as decorative categories or unlabeled color swatches.

### Cards / Containers

- **Corner Style:** Six-pixel record corners.
- **Background:** White sheets on the cool field.
- **Shadow Strategy:** None at rest.
- **Border:** One-pixel cool rule, with an internal rule under headings and between definition rows.
- **Internal Padding:** Most headings and dense facts use 16–18px; larger workflow sheets use 24px.

### Inputs / Fields

- **Style:** Cool-gray Carbon fields, full-width where the workflow requires it, with six-pixel top corners and a straight bottom border.
- **Focus:** Two-pixel blue inset outline.
- **Error / Disabled:** Errors appear in an adjacent explicit error notification; disabled controls retain readable labels and a visibly unavailable surface.

### Navigation

The desktop rail is a 232px graphite field with a 76px brand row. Navigation items are 46px high, use 13px inline padding, pair a 20px icon with a text label, and have six-pixel corners. Hover uses a slightly lighter graphite; the active destination uses a muted green fill and white text. Below 760px, the same navigation becomes a keyboard-managed drawer opened from the sticky mobile header.

### Page Header

Each application page starts with one condensed uppercase headline, one plain-language description, and at most one primary or supporting action. A strong rule below the header turns the title block into the top line of the inspection record.

### Metric Strip

Four equal facts share one bordered sheet. Each fact pairs a green 22px icon with a muted label and anchors a 37px tabular count at the bottom. Internal one-pixel dividers replace individual cards and adapt with the responsive column count.

### Event Ledger

The ledger uses 11px uppercase column labels on a cool-gray header, 13px body text, 59px rows, and horizontal scrolling when space is constrained. Type is reinforced with a small outlined square, numeric cells use tabular figures, review state is written in a tag, and the final column exposes a direct Inspect link.

### Notifications and Empty States

Low-contrast Carbon notifications span their container and use a six-pixel shell. They always include a semantic kind, title, and explanatory subtitle. Empty states center an icon, a concise heading, an evidence-bound explanation, and—when useful—a small tertiary upload action. Loading states reserve the final layout with gray shimmer blocks or explicit inline progress text.

## Do's and Don'ts

### Do:

- **Do** preserve the inspection-docket hierarchy: readiness, operational facts, evidence ledger, review work, and provenance.
- **Do** use white record sheets, cool-gray fields, one-pixel rules, and the six-pixel radius as the default component grammar.
- **Do** keep all numbers tabular and all scientific or review states explicit in text.
- **Do** provide real loading, empty, unavailable, warning, error, and recovery states before presenting live data.
- **Do** collapse split layouts at 1050px and replace the fixed rail with the accessible drawer at 760px.

### Don't:

- **Don't** fabricate event rows, evidence images, dates, distributions, readiness, approval, performance, or certificate marks.
- **Don't** turn the product into a neon AI command center, surveillance spectacle, or texture-heavy dashboard.
- **Don't** use green, amber, red, or blue as the only carrier of meaning.
- **Don't** add resting card shadows, oversized soft corners, ornamental gradients, or decorative glass effects.
- **Don't** compress evidence columns below legibility; preserve the ledger and allow horizontal scrolling.
