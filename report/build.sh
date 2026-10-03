#!/usr/bin/env bash
# Build the report: Mermaid -> SVG/PNG, Markdown -> HTML (pandoc) -> PDF (headless Chrome).
# Requirements: node/npx, pandoc >= 3, Google Chrome or Chromium.
#   CHROME=/path/to/chrome ./report/build.sh
set -euo pipefail
cd "$(dirname "$0")"

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PDF="../DDM501_Assignment2_[StudentID]_[Name].pdf"

# mermaid-cli drives a browser through puppeteer; reuse the local Chrome instead of a download.
PUPPETEER_CFG="$(mktemp)"
python3 - "$CHROME" "$PUPPETEER_CFG" <<'PY'
import json, sys
cfg = json.load(open("puppeteer-config.json"))
cfg["executablePath"] = sys.argv[1]
json.dump(cfg, open(sys.argv[2], "w"))
PY
export PUPPETEER_SKIP_DOWNLOAD=1
for src in diagrams/*.mmd; do
  base="${src%.mmd}"
  npx -y @mermaid-js/mermaid-cli@11 -q -p "$PUPPETEER_CFG" -c mermaid-config.json -b white -i "$src" -o "$base.svg"
  npx -y @mermaid-js/mermaid-cli@11 -q -p "$PUPPETEER_CFG" -c mermaid-config.json -b white -s 2 -i "$src" -o "$base.png"
done
rm -f "$PUPPETEER_CFG"

mkdir -p build
pandoc report.md --from markdown --to html5 --standalone --embed-resources \
  --css style.css --metadata pagetitle="DDM501 Assignment 2" -o build/report.html

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=5000 --print-to-pdf="$PDF" "file://$PWD/build/report.html" 2>/dev/null
echo "wrote $PDF"
