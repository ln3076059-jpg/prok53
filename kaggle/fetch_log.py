import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Fetching kernel log...")
try:
    log_content = api.kernels_logs("lethunga/dms-v2-baseline-evaluation")
    with open("kaggle_log.txt", "w", encoding="utf-8") as f:
        f.write(str(log_content))
    print("Done writing to kaggle_log.txt")
except Exception as e:
    print("Error:", e)
