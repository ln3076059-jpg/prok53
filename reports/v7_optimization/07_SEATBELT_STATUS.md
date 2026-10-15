# Seatbelt status

The existing 641-sample validation artifact remains unchanged: accuracy 80.97%, macro-F1 57.28%, balanced accuracy 57.22%, ECE 0.0846, and post-policy false-unfastened rate 10.29%.

The uncertain class has 25 verified samples: 1 correct and 24 failures. A review queue was generated at `reports\v7_optimization\seatbelt_uncertain_review_queue.csv`. ROI pixels are unavailable locally for all queued rows, so crop quality cannot be established and no labels or weights were changed.
