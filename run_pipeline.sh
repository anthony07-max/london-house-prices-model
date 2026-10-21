#!/usr/bin/env bash
set -e
python src/download.py   # ~550 MB of Price Paid Data is streamed and filtered to London (~40 MB kept)
python src/clean.py
python src/analysis.py
python src/model.py
