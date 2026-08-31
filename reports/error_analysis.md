# Error analysis

Status: **NOT_RUN**. Generate false-positive and false-negative galleries after the locked model has been evaluated.

Review phone errors for tiny or occluded phones, hands without phones, dashboard screens, mounts, and passenger phones. A passenger phone may be a correct detector box but must be a suppressed event. Review fastened errors for dark or partial belts, shadows, bag straps, and seams. Review unfastened errors for occlusion, crops, side views, unclear torsos, and people outside vehicles. Never reinterpret unclear belt visibility as ground truth for `seatbelt_unfastened`, and never count an observation without vehicle context as a violation.
