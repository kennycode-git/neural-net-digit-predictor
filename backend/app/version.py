import os

MODEL_VERSION = "1.0.0"
PREPROCESS_VERSION = "1"
# Injected at build/deploy time via GIT_COMMIT env var
GIT_COMMIT = os.environ.get("GIT_COMMIT", "dev")
