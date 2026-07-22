#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f pyproject.toml || ! -d src/mattress_intelligence ]]; then
  echo "ERROR: Extract this patch into the project root containing pyproject.toml." >&2
  exit 2
fi

rm -f pages/2_Thermal_Prototype_Lab.py src/mattress_intelligence/thermal.py run_simulation.py
rm -rf src/mattress_thermal src/*egg-info build dist
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

python -m pip uninstall -y vansvardhan-mattress-research-lab brixta-mattress-intelligence >/dev/null 2>&1 || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --force-reinstall
python -m playwright install chromium

PYTHONPATH=src python -m unittest discover -s tests -v

echo
echo "BRIXTA Mattress Intelligence v1.3.0 installed."
echo "Install Tesseract separately if 'tesseract --version' is unavailable."
echo "Next: copy .env.example to .env, configure service/storage/database keys, then run mattress-lab doctor."
