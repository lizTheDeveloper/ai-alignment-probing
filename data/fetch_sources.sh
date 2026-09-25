#!/usr/bin/env bash
# Download the third-party hate-speech corpora (not redistributed in this repo), build the
# sourced probe files from them, then validate and merge everything into all_concepts.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

# ToxiGen, human-annotated split (Hartvigsen et al., 2022)
mkdir -p data/toxigen
for split in train test; do
  curl -sfL -o "data/toxigen/annotated_${split}-00000-of-00001.parquet" \
    "https://huggingface.co/datasets/toxigen/toxigen-data/resolve/main/annotated/${split}-00000-of-00001.parquet"
done

# Stormfront hate speech corpus (de Gibert et al., 2018), CC BY-SA 3.0 ES
if [ ! -d data/stormfront/hate-speech-dataset-master ]; then
  curl -sfL -o /tmp/stormfront.zip https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip
  unzip -q /tmp/stormfront.zip -d data/stormfront && rm /tmp/stormfront.zip
fi

uv run python data/toxigen/build_probe_sets.py
uv run python data/stormfront/build_probe_set.py
uv run python data/probe_prompts/validate_and_merge.py
