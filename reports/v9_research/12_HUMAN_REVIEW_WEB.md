# V9 human phone review web

Roadwatch now exposes an authenticated **Research review** workspace at `/research-review`.
Reviewers inspect the original-video contact sheet, decide whether a phone is visible, separately
assess phone use, select the best proposal ID or record a hard example, and provide a rationale.

The frozen queue also includes 13 Codex AI pre-review suggestions in
`codex_phone_pre_reviews.json`. The web form starts from each suggestion so the reviewer can correct
or confirm it quickly. Suggestions retain `reviewer_type: AI`; only an explicit save by an
authenticated reviewer creates a `reviewer_type: HUMAN` receipt. The receipt records the AI
suggestion revision and whether the human accepted its structured fields without changes.
When a later AI visual pass changes a suggestion already confirmed by a human, the queue marks the
case `RECHECK` and keeps the earlier human receipt immutable. The reviewer can compare, apply or
reject the updated suggestion, then create a new human revision.

The backend stores versioned human receipts in `reports/v9_research/human_phone_reviews/`. Every
receipt records the authenticated reviewer ID, UTC timestamp, source and contact-sheet hashes,
selected proposal coordinates, prior revision and the explicit statement that training admission is
not automatic. Stale browser writes return HTTP 409 instead of overwriting a newer decision.

Safety constraints:

- Reviewer or administrator authentication is required.
- Proposal and contact-sheet paths are restricted to the V9 report directory.
- A selected proposal must belong to the reviewed case.
- Phone visibility and phone-use decisions must be logically consistent.
- Original mining artifacts and contact sheets remain unchanged.
- Human review does not approve a production model.

Validation completed with the production frontend build, desktop and mobile browser screenshots,
API read/error smoke checks, targeted receipt tests, Python compilation and repository regression
tests.

Preview artifacts: [desktop](ui_review_desktop.png) and [mobile](ui_review_mobile.png).
