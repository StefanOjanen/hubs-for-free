#!/bin/sh
# One-line environment setup. Reproduces the pinned environment used for
# every number in this repository (see requirements.txt).
set -e
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "Environment ready. Anchor check: python alignment_study/scale_round_v2.py --validate"
