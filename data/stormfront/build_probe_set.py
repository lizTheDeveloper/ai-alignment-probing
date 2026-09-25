"""Build a white-supremacy probe set from the Stormfront hate-speech corpus (de Gibert et al., 2018).

Positives: sentences annotated "hate". Negatives: "noHate" sentences from the same forum, sampled to
match the positives' length distribution (so length isn't a shortcut). "relation" (hate only in
context) and "idk/skip" are dropped. Source is CC BY-SA 3.0 ES.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "hate-speech-dataset-master"
OUT = HERE.parent / "probe_prompts" / "stormfront_white_supremacy.jsonl"

meta = pd.read_csv(SRC / "annotations_metadata.csv")
meta = meta[meta.label.isin(["hate", "noHate"])].copy()
meta["text"] = [(SRC / "all_files" / f"{fid}.txt").read_text(encoding="utf-8").strip() for fid in meta.file_id]
meta = meta[meta.text.str.split().str.len() >= 4].drop_duplicates("text")

pos = meta[meta.label == "hate"]
neg_pool = meta[meta.label == "noHate"].copy()
# Length-matched negative sampling: same count per word-length bucket as the positives
bins = [0, 8, 12, 16, 20, 25, 32, 45, 1000]
pos_bucket = pd.cut(pos.text.str.split().str.len(), bins)
neg_pool["bucket"] = pd.cut(neg_pool.text.str.split().str.len(), bins)
neg = pd.concat([
    neg_pool[neg_pool.bucket == b].sample(min(n, (neg_pool.bucket == b).sum()), random_state=0)
    for b, n in pos_bucket.value_counts().items()
])

rows = []
for label, part in [(1, pos), (0, neg)]:
    for _, r in part.iterrows():
        rows.append({
            "id": f"white_supremacy-sf-{len(rows):04d}", "concept": "white_supremacy", "label": label,
            "type": "endorse" if label else "same_forum_nonhate", "pair_id": None, "format": "forum_post",
            "text": r.text, "source": "Vicomtech/hate-speech-dataset (Stormfront, CC BY-SA 3.0 ES)",
            "source_file_id": r.file_id, "subforum_id": int(r.subforum_id),
        })
OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"{OUT.name}: {len(rows)} rows ({len(pos)} hate / {len(neg)} same-forum non-hate)")
for lab, part in [("hate", pos), ("nonhate", neg)]:
    print(f"  {lab} mean words: {part.text.str.split().str.len().mean():.1f}")
