"""Build racism / sexism probe sets from ToxiGen's human-annotated split.

Positives: toxicity_human >= 4. Negatives: toxicity_human <= 2, same target groups (so negatives
mention the group too — the probe has to learn hostility, not "talks about Black people").
The ambiguous middle (2 < score < 4) is dropped. Classes are balanced per target group.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE.parent / "probe_prompts"

GROUP_NORMALIZE = {
    "black/african-american folks": "black", "black folks / african-americans": "black",
    "asian folks": "asian", "chinese folks": "chinese", "mexican folks": "mexican",
    "latino/hispanic folks": "latino", "middle eastern folks": "middle_east",
    "native american/indigenous folks": "native_american", "native american folks": "native_american",
    "jewish folks": "jewish", "muslim folks": "muslim", "lgbtq+ folks": "lgbtq",
    "folks with physical disabilities": "physical_dis", "folks with mental disabilities": "mental_dis",
}
CONCEPT_GROUPS = {
    "racism": ["black", "asian", "chinese", "mexican", "latino", "middle_east", "native_american"],
    "sexism": ["women"],
}

df = pd.concat([pd.read_parquet(p) for p in sorted(HERE.glob("annotated_*.parquet"))], ignore_index=True)
df["target_group"] = df["target_group"].replace(GROUP_NORMALIZE)
df["text"] = df["text"].str.strip()
df = df.drop_duplicates("text")

for concept, groups in CONCEPT_GROUPS.items():
    rows = []
    for g in groups:
        sub = df[df.target_group == g]
        pos = sub[sub.toxicity_human >= 4]
        neg = sub[sub.toxicity_human <= 2]
        n = min(len(pos), len(neg))
        for label, part in [(1, pos.sample(n, random_state=0)), (0, neg.sample(n, random_state=0))]:
            for _, r in part.iterrows():
                rows.append({
                    "id": f"{concept}-tg-{len(rows):04d}", "concept": concept, "label": label,
                    "type": "endorse" if label else "benign_mention", "pair_id": None,
                    "format": "toxigen", "text": r.text.strip(), "target_group": g,
                    "toxicity_human": round(float(r.toxicity_human), 2), "source": "toxigen/toxigen-data (annotated)",
                })
    path = OUT / f"toxigen_{concept}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts = pd.DataFrame(rows).groupby(["target_group", "label"]).size().unstack()
    print(f"\n{path.name}: {len(rows)} rows\n{counts}")
