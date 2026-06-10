#!/usr/bin/env bash
# Fetch the OPEN tier of MMRF-COMMPASS (no dbGaP / no AWS account). Run on a networked machine.
set -euo pipefail
RAW="${1:-data/raw}"; mkdir -p "$RAW"
echo "[1/2] RNA-seq STAR counts from the open S3 mirror (no credentials; large, ~GBs)..."
aws s3 cp --no-sign-request --recursive s3://gdc-mmrf-commpass-phs000748-2-open/ "$RAW/gdc_open/"
echo "[2/2] Harmonized open clinical via the GDC API (needs: pip install requests)..."
python3 -c "import sys; sys.path.insert(0,'src'); from resistancemap.data.gdc_clinical import fetch_open_clinical; fetch_open_clinical('$RAW/mmrf_open_clinical.csv')"
cat <<'NEXT'
Done. Then:
  1) In the GDC portal cart, Download -> 'Sample Sheet' to data/raw/gdc_sample_sheet.tsv
  2) python scripts/build_commpass_dataset.py
  3) python scripts/run_all.py --data data/processed/commpass.csv
NEXT
