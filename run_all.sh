#!/usr/bin/env bash
# End-to-end: synthetic data -> features -> model -> offers -> dashboard. ~1.5 min on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/gen_data.py data
python3 src/run_pipeline.py data out
python3 src/build_dashboard.py
echo "Open dashboard/index.html in a browser."
