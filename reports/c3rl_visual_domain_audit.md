# c3rl seatbelt visual-domain spot audit

Date: 2026-08-31

Decision: **AUXILIARY_SYNTHETIC_ONLY_PROVENANCE_REVIEW_REQUIRED**

Four files were visually inspected at original resolution, covering both folder classes and including randomly selected assets:

- `no_seatbelt/0042b9e2-ad01-4d79-9e8a-fd2e7b51415a.jpg`
- `seatbelt/0003b537-d9c9-457c-8fd6-926dcd8d675c.jpg`
- `no_seatbelt/ffc275df-6427-453e-ae31-8344808563ca.jpg`
- `seatbelt/399d8a07-0d0d-4754-b9c0-7c7a7c86b4d7.jpg`

All four show geometric synthetic person/vehicle scenes rather than photographic cabin occupants. This is a spot audit, not a claim that every file was visually examined. It is sufficient to reject the previous assumption that this source can serve as the main real-camera seatbelt dataset. The 5,000 raw files remain immutable for reproducibility, but they must not enter governed real-camera train/validation/test data. Any future use must be an explicitly separated synthetic smoke test or ablation.
