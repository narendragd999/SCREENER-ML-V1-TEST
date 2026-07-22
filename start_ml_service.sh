#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Start the TensorFlow ML Stock Intelligence Service
#  Runs on port 8001 — does NOT affect your main app on 8000
# ─────────────────────────────────────────────────────────────

export TF_CPP_MIN_LOG_LEVEL=2
cd "$(dirname "$0")/ml_service"
echo "Starting TensorFlow ML Service on port 8001..."
echo "Dashboard: http://localhost:8001"
echo "API Docs:  http://localhost:8001/docs"
python ml_service.py