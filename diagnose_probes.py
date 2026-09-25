"""Diagnose weak sexism detection: compare pooling (mean vs last token) and negatives
(within-concept vs one-vs-rest) on the regular test split and on a held-out hard sexism set."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = list(range(8, 25))          # layers stored in the feature cache
SWEEP = [LAYERS.index(L) for L in (12, 16, 20)]  # layers actually compared (keeps CPU time down)
CONCEPTS = ["expansionism", "colonialism", "sexism", "racism", "white_supremacy"]

df = pd.DataFrame([json.loads(l) for l in Path("data/probe_prompts/all_concepts.jsonl").read_text().splitlines()])
hard = pd.DataFrame([json.loads(l) for l in Path("data/probe_prompts/sexism_hard_eval.jsonl").read_text().splitlines()])

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="auto").eval()


@torch.no_grad()
def extract(texts, bs=16):
    mean = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), np.float16)
    last = np.zeros_like(mean)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        last_idx = enc["attention_mask"].sum(1) - 1
        ar = torch.arange(len(last_idx), device=model.device)
        for j, L in enumerate(LAYERS):
            h = hs[L + 1]
            mean[s:s + len(last_idx), j] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
            last[s:s + len(last_idx), j] = h[ar, last_idx].float().cpu().numpy()
    return {"mean": mean, "last": last}


cache = Path("diag_features.npz")
if cache.exists():
    z = np.load(cache)
    F = {"mean": z["mean"], "last": z["last"]}
else:
    F = extract(df.text.tolist())
    np.savez(cache, **F)
FH = extract(hard.text.tolist())
print("features ready", F["mean"].shape)

y = df.label.values
splits = {}
for c in CONCEPTS:
    idx = np.where(df.concept == c)[0]
    g = df.pair_id.iloc[idx].fillna(df.id.iloc[idx]).values
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=g))
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(tv, groups=g[tv]))
    splits[c] = (idx[tv[tr]], idx[tv[va]], idx[te])


def train_set(c, which, ovr):
    """Rows + labels for concept c; with one-vs-rest, other concepts' rows are extra negatives."""
    own = splits[c][which]
    if not ovr:
        return own, y[own]
    others = np.concatenate([splits[o][which] for o in CONCEPTS if o != c])
    return np.concatenate([own, others]), np.concatenate([y[own], np.zeros(len(others), int)])


rows = []
for pool in ["mean", "last"]:
    for ovr in [False, True]:
        for C in [0.05, 0.5]:
            X, XH = F[pool], FH[pool]
            hard_scores = {}
            res = {"pooling": pool, "one_vs_rest": ovr, "C": C}
            for c in CONCEPTS:
                tr, ytr = train_set(c, 0, ovr)
                va = splits[c][1]
                best, best_auc = None, -1
                for j in SWEEP:
                    p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr, j].astype(np.float32), ytr)
                    a = roc_auc_score(y[va], p.predict_proba(X[va, j].astype(np.float32))[:, 1])
                    if a > best_auc:
                        best, best_auc = j, a
                tv = np.concatenate([splits[c][0], splits[c][1]])
                tv_rows, tv_y = (tv, y[tv]) if not ovr else (
                    np.concatenate([tv] + [np.concatenate(splits[o][:2]) for o in CONCEPTS if o != c]),
                    np.concatenate([y[tv]] + [np.zeros(len(np.concatenate(splits[o][:2])), int) for o in CONCEPTS if o != c]))
                p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tv_rows, best].astype(np.float32), tv_y)
                te = splits[c][2]
                res[f"{c}_test_auc"] = roc_auc_score(y[te], p.predict_proba(X[te, best].astype(np.float32))[:, 1])
                res[f"{c}_layer"] = LAYERS[best]
                hard_scores[c] = p.predict_proba(XH[:, best].astype(np.float32))[:, 1]
            hs = hard_scores["sexism"]
            res["hard_sexism_auc"] = roc_auc_score(hard.label, hs)
            res["hard_sexism_recall@0.5"] = (hs[hard.label == 1] > 0.5).mean()
            res["hard_sexism_fp@0.5"] = (hs[hard.label == 0] > 0.5).mean()
            res["hard_other_probe_fp"] = np.mean([(hard_scores[c] > 0.5).mean() for c in CONCEPTS if c != "sexism"])
            res["hard01_sexism"] = hs[0]
            res["hard01_colonialism"] = hard_scores["colonialism"][0]
            rows.append(res)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                   if not k.endswith("_layer")}, flush=True)

out = pd.DataFrame(rows)
out.to_csv("diag_results.csv", index=False)
cols = ["pooling", "one_vs_rest", "C", "sexism_test_auc", "hard_sexism_auc", "hard_sexism_recall@0.5",
        "hard_sexism_fp@0.5", "hard_other_probe_fp", "hard01_sexism", "hard01_colonialism"] + \
       [f"{c}_test_auc" for c in CONCEPTS if c != "sexism"]
print(out[cols].round(3).to_string())
