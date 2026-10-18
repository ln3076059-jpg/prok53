# V14 driver association evidence

The autonomous adapter transports runtime driver-region provenance, role, track identity, confidence and verification status. An absent driver region falls back to the full source frame for candidate preservation but keeps role state UNKNOWN. Passenger evidence remains passenger evidence; a visible phone alone is never a driver violation.

Existing association logic continues to abstain for overlapping occupants, unverified tracks, unsupported roles and unverified mounted/static context. Synthetic tests cover passengers, ambiguity and missing evidence. Real association accuracy is not reported because the 13 development FNs contain zero verified occupant tracks and there is no association ground-truth benchmark.
