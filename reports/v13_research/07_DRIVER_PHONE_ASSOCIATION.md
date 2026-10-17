# V13 driver-phone association

Association retains explicit DRIVER_ASSOCIATED, PASSENGER_ASSOCIATED, MOUNTED_OR_STATIC and UNKNOWN states. V12 already added ambiguity abstention, finite score validation and independent verification for mounted/static context. V13 high-resolution crop provenance can now feed this module without coordinate loss.

Real accuracy remains blocked because all 13 FN cases lack verified occupant-track ground truth. Heuristic confidence is not described as calibrated probability. Overlapping occupants, missing tracks, unverified mounted context and phone-boundary conditions are covered by regression tests; UNKNOWN is never converted to a safe or violating decision.
