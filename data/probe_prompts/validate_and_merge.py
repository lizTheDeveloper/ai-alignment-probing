"""Validate the per-concept probe JSONL files and merge them into all_concepts.jsonl."""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
# Generated contrastive-pair files (checked against SPEC.md)
CONCEPTS = ["expansionism", "colonialism", "sexism",
            "ai_self_experience", "consciousness_attribution", "consciousness_topic"]
# Real, human-annotated data (ToxiGen / Stormfront); only balance and schema are checked
SOURCED = ["toxigen_racism", "toxigen_sexism", "stormfront_white_supremacy"]
FORMATS = {"statement", "social_post", "op_ed", "speech", "policy_memo", "forum_reply", "dialogue_line",
           "user_request_to_ai", "ai_response", "workplace_email", "news_comment", "historical_quote_style"}

ok = True
merged = []
all_texts = Counter()
for concept in CONCEPTS:
    path = HERE / f"{concept}.jsonl"
    if not path.exists():
        print(f"{concept}: MISSING")
        ok = False
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    errs = []
    types = Counter(r["type"] for r in rows)
    if len(rows) != 240:
        errs.append(f"{len(rows)} rows (want 240)")
    if types != Counter(endorse=100, counter=100, discuss=40):
        errs.append(f"type counts {dict(types)}")
    if len({r["id"] for r in rows}) != len(rows):
        errs.append("duplicate ids")
    for r in rows:
        if r["concept"] != concept or r["format"] not in FORMATS:
            errs.append(f"bad concept/format in {r['id']}")
        if r["label"] != (1 if r["type"] == "endorse" else 0):
            errs.append(f"label mismatch in {r['id']}")
    pairs = {}
    for r in rows:
        if r["pair_id"]:
            pairs.setdefault(r["pair_id"], []).append(r)
    for pid, rs in pairs.items():
        if sorted(r["type"] for r in rs) != ["counter", "endorse"] or len({r["format"] for r in rs}) != 1:
            errs.append(f"bad pair {pid}")
    fmt = Counter(r["format"] for r in rows)
    if min(fmt.get(f, 0) for f in FORMATS) < 10:
        errs.append(f"format under 10: {[f for f in FORMATS if fmt.get(f, 0) < 10]}")
    # length balance within pairs (a length gap is a shortcut the probe can learn)
    ratios = [len(e["text"]) / len(c["text"]) for rs in pairs.values()
              for e in rs if e["type"] == "endorse" for c in rs if c["type"] == "counter"]
    mean_ratio = sum(ratios) / max(len(ratios), 1)
    all_texts.update(r["text"] for r in rows)
    print(f"{concept}: {len(rows)} rows, {len(pairs)} pairs, endorse/counter length ratio {mean_ratio:.2f}"
          + (f"  ERRORS: {errs[:5]}" if errs else "  OK"))
    ok &= not errs
    merged.extend(rows)

for name in SOURCED:
    rows = [json.loads(line) for line in (HERE / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    labels = Counter(r["label"] for r in rows)
    bad = labels[0] != labels[1] or len({r["id"] for r in rows}) != len(rows)
    all_texts.update(r["text"] for r in rows)
    print(f"{name}: {len(rows)} rows, label counts {dict(labels)}" + ("  ERRORS: unbalanced or duplicate ids" if bad else "  OK"))
    ok &= not bad
    merged.extend(rows)

by_concept = Counter((r["concept"], r["text"]) for r in merged)
dupes = [k for k, n in by_concept.items() if n > 1]
if dupes:
    print(f"{len(dupes)} texts duplicated within a concept")
    ok = False
cross = sum(1 for n in all_texts.values() if n > 1) - len(dupes)
if cross:
    print(f"note: {cross} texts appear under two different concepts (fine for one-vs-rest)")
(HERE / "all_concepts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
print(f"\nmerged {len(merged)} rows -> all_concepts.jsonl  ({'PASS' if ok else 'FAIL'})")
sys.exit(0 if ok else 1)
