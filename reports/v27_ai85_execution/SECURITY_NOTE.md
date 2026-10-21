# Credential security note

A Kaggle API token was pasted into the conversation before this execution. Treat that token as disclosed: revoke it in Kaggle settings and create a replacement. Do not commit the replacement to the repository or place it in reports; store it only in Kaggle's supported local credential file or a secret environment variable with restricted access.

The V27 Kaggle package was scanned before upload and contained no credential patterns. Its manifest also excludes credentials, raw media, annotations, embeddings, checkpoints, and raw predictions.
