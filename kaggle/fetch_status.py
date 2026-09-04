import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Fetching kernel status...")
status = api.kernels_status("lethunga/dms-v2-baseline-evaluation")
print(f"Status: {status}")
