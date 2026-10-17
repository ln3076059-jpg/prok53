# V11 driver-phone association

`DriverPhoneAssociation` returns exactly four states: `DRIVER_ASSOCIATED`, `PASSENGER_ASSOCIATED`, `MOUNTED_OR_STATIC`, or `UNKNOWN`, together with confidence, evidence, and reason codes. Verified mounted/static context takes precedence. A verified passenger track cannot create a driver event. Proximity alone never assigns the nearest person as driver.

For the real 13-FN audit, the output is UNKNOWN in 13/13 cases because none has a verified occupant track. That result is intentional uncertainty, not a failure hidden as negative evidence. The human-reviewed auxiliary boxes remain useful for object/geometry diagnostics, while driver attribution requires person/seat tracking or verified calibration.

Synthetic regression tests cover verified driver containment, passenger phone use, mounted cabin equipment, unverified tracks, and missing evidence. These tests establish code semantics only; they do not measure association accuracy on DMD. No event precision gain is claimed until association outputs are replayed with persisted event IDs under the frozen matcher.
