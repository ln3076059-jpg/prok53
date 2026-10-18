# Dataset research and licenses

Nine requested sources were checked against official owner pages on 2026-10-18. A public URL
was treated as discovery evidence only. No V16 media download occurred.

- **DMD 2026:** the owner lists RGB real-car data for 14 subjects, OpenLABEL/VCD 5+ labels and
  CC BY-NC-ND 4.0. The project permission record still says `RIGHTS_ACCEPTED=false`; processing
  or fitting permissions were not changed. `gZ-37` and `gE-28` remain protected.
- **100-Driver:** the official page reports more than 470,000 images, 100 drivers, five vehicles
  and four cameras. The official custom form requires prior PIC Lab consent, research-only use
  and no redistribution. Repository code being MIT does not license its data.
- **State Farm:** access remains subject to Kaggle competition rules and account acceptance.
  The existing local registry also rejects it pending legal review. No credential or API token
  was used.
- **Drive&Act:** the official page reports 12 hours, 29 long sequences, five views and 83
  hierarchical activities under research-use conditions. A phone-equivalent label was not
  established, so no automatic PHONE_USE mapping is allowed.
- **COCO 2017:** useful for generic cell-phone boxes, but image-level rights and the selected
  subset still require verification. The full archive was deliberately not downloaded.
- **Open Images V7:** official metadata identifies `/m/050k8` as `Mobile phone`. Verified
  negatives and per-image licenses are required; missing boxes alone do not make a negative.
- **Roboflow candidates:** all three pages display CC BY 4.0, but original-image provenance,
  augmentation lineage and label meaning remain unresolved. The Eco3cc page does not prove a
  consistent exported UNCERTAIN class.

User actions needed before access changes: submit the official 100-Driver license form; accept
the State Farm competition rules through the user's own Kaggle account; provide an authorized
DMD rights record; and complete provenance/label review for Roboflow assets. Credentials must
not be committed or pasted into reports.

Authorization procedure:

1. For 100-Driver, download the official licensing PDF, describe this non-commercial research,
   sign it with the responsible institution/researcher, send it to the PIC Lab contact printed
   on the form, and retain their written approval reference. Only then may a human set the local
   registry to `DOWNLOAD_AUTHORIZED` with `allowed_operations: [download]`.
2. For State Farm, the user must sign into Kaggle and accept the competition rules personally.
   Record the accepted rules/version and permitted operations; store the token only in the
   environment. Do not put a token in YAML, a command transcript, Git or this report.
3. For DMD, an authorized human must update the separate project permission record. Dataset
   access from another source cannot alter `RIGHTS_ACCEPTED` or authorize model fitting.
4. For each Roboflow candidate, sample original assets, trace their origin/license and inspect
   exact exported classes before allowing ingestion beyond quarantine. Public project licensing
   alone does not approve the assets for Roadwatch training.
