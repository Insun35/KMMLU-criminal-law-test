#!/usr/bin/env bash
set -euo pipefail

echo "🔄 1) Load law IDs and save to data/raw/law_ids.json"
poetry run python -m scripts.load_data

echo "🔄 2) Prepare raw statutes & merge into JSONL"
poetry run python -m scripts.prepare_data

echo "🔄 3) Build evaluation batch input JSONL"
poetry run python -m scripts.build_batch_input

echo "🔄 4) Submit & poll & download results & compute accuracy"
poetry run python -m scripts.evaluate

echo "✅ All done! See score.txt for the final accuracy."
