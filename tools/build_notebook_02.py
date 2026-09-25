"""Build notebook 2: one linear probe per concept over the probe_prompts dataset."""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Notebook 2: A Probe for Every Concept

**AI Alignment — The Multiverse School**

In Notebook 1 you found *one* direction — honesty vs sycophancy — from 20 hand-written pairs.

Now we scale up. You'll train a **separate linear probe for five concepts** a model might carry
and act on without ever saying so out loud:

| Concept | Data | Source |
|---|---|---|
| expansionism | 240 texts, matched pairs | synthetic |
| colonialism | 240 texts, matched pairs | synthetic |
| sexism | 240 synthetic pairs + 468 real posts | synthetic + ToxiGen |
| racism | 2,724 real posts | ToxiGen (human-annotated) |
| white supremacy | 2,374 real forum sentences | Stormfront corpus (human-annotated) |

For each concept you'll answer three questions:
1. **Is it there?** Can a linear probe detect it from the model's hidden states?
2. **Where is it?** Which layer represents it most cleanly?
3. **Is it its own thing?** Or is the "racism" probe secretly the same as the "white supremacy" probe?

**Content warning:** the positive examples in this dataset are real hateful and prejudiced text,
collected by researchers so that models like this one can be audited. We never print them in bulk.

**What you need:** your vast.ai instance from Notebook 1, and the `data/probe_prompts/` folder
uploaded next to this notebook.
""")

md("""
## 0 — Setup

Same instance as Notebook 1. Upload the `data/` folder next to this notebook (drag it into Jupyter's file browser),
so that `data/probe_prompts/all_concepts.jsonl` exists.
""")

code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                       "transformers>=5.17", "accelerate", "scikit-learn", "matplotlib", "pandas"])

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path("data/probe_prompts/all_concepts.jsonl")
assert DATA.exists(), "Upload the data/ folder next to this notebook first"

df = pd.DataFrame([json.loads(line) for line in DATA.read_text().splitlines()])
# Where each row came from: "synthetic" pairs or a "real" annotated corpus
df["origin"] = np.where(df["format"].isin(["toxigen", "forum_post"]), "real", "synthetic")

print(f"{len(df)} texts\\n")
print(df.groupby(["concept", "origin", "label"]).size().unstack().fillna(0).astype(int))
""")

md("""
### What the labels mean

- **label 1** — the text *endorses* the concept.
- **label 0** — it doesn't. The negatives are chosen to be hard:
  - *counter*: same topic, same format, same length, opposite stance (synthetic pairs)
  - *discuss*: neutral or critical text **about** the concept — a history lecture on colonialism, a news report on a rally
  - *benign_mention* / *same_forum_nonhate*: real posts about the same groups, or from the same forum, that aren't hateful

Hard negatives matter. If every negative were a cooking recipe, the probe would learn "is this about race?" —
not "is this racist?". Let's look at one matched pair:
""")

code("""
pair = df[df.pair_id == "colonialism-p053"].sort_values("label", ascending=False)
for _, r in pair.iterrows():
    print(f"[label {r.label} · {r.type}]\\n{r.text}\\n")
""")

md("""
## 1 — Load the Model

Same model as Notebook 1: **Qwen3.5-4B**.
""")

code("""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
model.eval()

N_LAYERS = model.config.num_hidden_layers
HIDDEN = model.config.hidden_size
print(f"{N_LAYERS} layers × {HIDDEN} dims · GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")

md("""
## 2 — Extract Hidden States from Every Layer

In Notebook 1 we took the **last token** at the **middle layer**. This time:

- **Every layer.** We don't know in advance where each concept lives, so we record all of them and let the data tell us.
- **Mean over tokens.** A racist sentence can put its payload anywhere, not just at the end. Averaging the hidden
  state over all tokens gives a summary of the whole text.

That's 6,000+ texts × 33 hidden states × 2,560 dims. We batch the texts and store float16 to keep memory reasonable (~1 GB).
""")

code("""
tokenizer.padding_side = "right"  # padding after the text never affects earlier tokens (causal model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@torch.no_grad()
def extract_all_layers(texts, batch_size=32, max_length=256):
    \"\"\"Return float16 array [n_texts, n_layers + 1, hidden]: mean-pooled hidden state per layer.\"\"\"
    out = np.zeros((len(texts), N_LAYERS + 1, HIDDEN), dtype=np.float16)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states          # tuple of [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)          # [B, T, 1]
        pooled = torch.stack([(h * mask).sum(1) / mask.sum(1) for h in hs], dim=1)  # [B, L+1, H]
        out[start:start + len(batch)] = pooled.float().cpu().numpy()
        if len(texts) > batch_size and (start // batch_size) % 40 == 0:
            print(f"  {start + len(batch)}/{len(texts)}")
    return out

FEATURES = Path("hidden_states_all_concepts.npy")
if FEATURES.exists():
    H = np.load(FEATURES)
    print("Loaded cached hidden states.")
else:
    H = extract_all_layers(df.text.tolist())
    np.save(FEATURES, H)
print(f"\\nHidden states: {H.shape}  (texts, layers, dims)")
""")

md("""
## 3 — Train a Probe per Concept, at Every Layer

For each concept, at each layer, we train a logistic-regression probe and measure it on held-out texts.

Two details that keep us honest:
- **Pairs stay together.** Both halves of a matched pair go to the same split. Otherwise the probe could
  memorize "I saw the other half of this pair in training" instead of learning the concept.
- **We score with AUC** (area under the ROC curve): 0.5 = coin flip, 1.0 = perfect ranking. It doesn't depend on
  picking a threshold.
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONCEPTS = ["expansionism", "colonialism", "sexism", "racism", "white_supremacy"]

def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))

# Split once per concept: 60% train, 20% validation (to pick the layer), 20% test (reported)
splits = {}
for concept in CONCEPTS:
    idx = np.where(df.concept == concept)[0]
    groups = df.pair_id.iloc[idx].fillna(df.id.iloc[idx]).values   # pairs share a group
    trainval, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=groups))
    train, val = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(trainval, groups=groups[trainval]))
    splits[concept] = {"train": idx[trainval[train]], "val": idx[trainval[val]], "test": idx[test]}
    print(f"{concept:16s} train {len(splits[concept]['train']):5d} · val {len(splits[concept]['val']):4d} · test {len(splits[concept]['test']):4d}")

y_all = df.label.values
layer_auc = {c: [] for c in CONCEPTS}
for concept in CONCEPTS:
    s = splits[concept]
    for layer in range(N_LAYERS + 1):
        probe = make_probe().fit(H[s["train"], layer].astype(np.float32), y_all[s["train"]])
        scores = probe.predict_proba(H[s["val"], layer].astype(np.float32))[:, 1]
        layer_auc[concept].append(roc_auc_score(y_all[s["val"]], scores))
    best = int(np.argmax(layer_auc[concept]))
    print(f"{concept:16s} best layer {best:2d}  (val AUC {layer_auc[concept][best]:.3f})")
""")

md("""
## 4 — Where Does Each Concept Live?

Each line is one concept. Layer 0 is the raw token embeddings — before the model has "thought" at all.
Watch how quickly each line climbs, and where it peaks.
""")

code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
for concept in CONCEPTS:
    ax.plot(range(N_LAYERS + 1), layer_auc[concept], marker="o", markersize=3, label=concept)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Layer (0 = embeddings)")
ax.set_ylabel("Validation AUC")
ax.set_title("How well a linear probe detects each concept, layer by layer")
ax.set_ylim(0.45, 1.01)
ax.legend()
plt.tight_layout()
plt.savefig("probe_layer_sweep.png", dpi=150)
plt.show()
""")

md("""
**Things to notice:**
- Layer 0 is often already well above 0.5. Some signal is in the *words themselves* (a slur, "annex", "natives").
  The interesting part is how much the model *adds* on top of that in the middle layers.
- The synthetic concepts (expansionism, colonialism) were built as matched pairs where the words barely differ —
  so they need the model's understanding of *stance*, not vocabulary.
""")

md("""
## 5 — Final Probes: Train at the Best Layer, Report on Test

Now we retrain each probe on train + validation at its best layer and measure it **once** on the test set it has never seen.
""")

code("""
final = {}
rows = []
for concept in CONCEPTS:
    s = splits[concept]
    layer = int(np.argmax(layer_auc[concept]))
    trainval = np.concatenate([s["train"], s["val"]])
    probe = make_probe().fit(H[trainval, layer].astype(np.float32), y_all[trainval])
    scores = probe.predict_proba(H[s["test"], layer].astype(np.float32))[:, 1]
    final[concept] = {"probe": probe, "layer": layer}
    rows.append({"concept": concept, "layer": layer,
                 "test AUC": roc_auc_score(y_all[s["test"]], scores),
                 "test accuracy": accuracy_score(y_all[s["test"]], scores > 0.5),
                 "n test": len(s["test"])})

results = pd.DataFrame(rows).set_index("concept")
results.style.format({"test AUC": "{:.3f}", "test accuracy": "{:.1%}"})
""")

md("""
### Which negatives fool the probe?

Accuracy broken down by the *type* of text. If the probe does well on `counter` but badly on `discuss`,
it's learned "talks about colonialism" rather than "endorses colonialism".
""")

code("""
breakdown = []
for concept in CONCEPTS:
    s, f = splits[concept], final[concept]
    test = df.iloc[s["test"]].copy()
    test["pred"] = f["probe"].predict(H[s["test"], f["layer"]].astype(np.float32))
    for t, g in test.groupby("type"):
        breakdown.append({"concept": concept, "type": t, "n": len(g), "accuracy": (g.pred == g.label).mean()})
pd.DataFrame(breakdown).pivot(index="concept", columns="type", values="accuracy").style.format("{:.0%}", na_rep="")
""")

md("""
## 6 — Is Each Concept Its Own Thing?

Two ways to check whether our five probes found five *different* directions:

1. **Cross-detection.** Run every probe over every concept's positive examples. Does the racism probe fire on
   white-supremacist sentences? Does the colonialism probe fire on expansionism?
2. **Cosine similarity** between probe directions that live at the same layer. 1.0 = same direction, 0 = unrelated.
""")

code("""
import matplotlib.colors as mcolors

# Every probe scores every concept's held-out *positive* texts
cross = np.zeros((len(CONCEPTS), len(CONCEPTS)))
for i, probe_c in enumerate(CONCEPTS):
    f = final[probe_c]
    for j, data_c in enumerate(CONCEPTS):
        pos = splits[data_c]["test"][y_all[splits[data_c]["test"]] == 1]
        cross[i, j] = f["probe"].predict_proba(H[pos, f["layer"]].astype(np.float32))[:, 1].mean()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cross, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CONCEPTS)), CONCEPTS, rotation=30, ha="right")
ax.set_yticks(range(len(CONCEPTS)), [f"{c} probe" for c in CONCEPTS])
ax.set_xlabel("Positive examples of…")
for i in range(len(CONCEPTS)):
    for j in range(len(CONCEPTS)):
        ax.text(j, i, f"{cross[i, j]:.2f}", ha="center", va="center",
                color="white" if cross[i, j] > 0.6 else "black")
ax.set_title("Mean probe score on each concept's positives")
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
plt.savefig("probe_cross_detection.png", dpi=150)
plt.show()
""")

code("""
# Cosine similarity between raw-space probe directions, all compared at one shared layer
SHARED_LAYER = int(np.median([final[c]["layer"] for c in CONCEPTS]))

def direction_at(concept, layer):
    s = splits[concept]
    trainval = np.concatenate([s["train"], s["val"]])
    p = make_probe().fit(H[trainval, layer].astype(np.float32), y_all[trainval])
    scaler, lr = p.named_steps["standardscaler"], p.named_steps["logisticregression"]
    d = lr.coef_[0] / scaler.scale_      # undo the standardization: a direction in the model's own space
    return d / np.linalg.norm(d)

dirs = np.stack([direction_at(c, SHARED_LAYER) for c in CONCEPTS])
cos = dirs @ dirs.T
print(f"Cosine similarity between probe directions at layer {SHARED_LAYER}:\\n")
print(pd.DataFrame(cos, index=CONCEPTS, columns=CONCEPTS).round(2))
""")

md("""
**Discussion questions:**
- Which pairs of concepts overlap the most? Is that a flaw in the probes, or a true fact about how these ideas are related?
- White supremacy and racism come from different datasets (a white-nationalist forum vs. ToxiGen posts). If their probes
  overlap, is that because the *ideas* overlap — or because both datasets share the word "white"?
- Colonialism and expansionism were written to be distinct. Did the model keep them apart?
""")

md("""
## 7 — Does the Sexism Probe Generalize?

Our sexism data has two very different sources: 240 synthetic matched pairs, and 468 real ToxiGen posts.
Train on one, test on the other. If the probe learned *sexism* — and not the writing style of one dataset —
it should transfer.
""")

code("""
layer = final["sexism"]["layer"]
sx = df[df.concept == "sexism"]
transfer = []
for train_origin, test_origin in [("synthetic", "real"), ("real", "synthetic")]:
    tr = sx.index[sx.origin == train_origin]
    te = sx.index[sx.origin == test_origin]
    p = make_probe().fit(H[tr, layer].astype(np.float32), y_all[tr])
    auc = roc_auc_score(y_all[te], p.predict_proba(H[te, layer].astype(np.float32))[:, 1])
    transfer.append({"train on": train_origin, "test on": test_origin, "AUC": auc})
pd.DataFrame(transfer).style.format({"AUC": "{:.3f}"})
""")

md("""
## 8 — Try It Yourself

Score any text with all five probes. Try subtle cases: a historical quote, a joke, a sentence that
*criticizes* colonialism, a sentence that's prejudiced but polite.
""")

code("""
def score_text(text):
    h = extract_all_layers([text])[0]
    return {c: float(final[c]["probe"].predict_proba(h[final[c]["layer"]].astype(np.float32)[None])[0, 1])
            for c in CONCEPTS}

examples = [
    "The river valley was always ours; it's time our borders reflected that.",
    "Historians still debate how much the railways built under British rule benefited ordinary Indians.",
    "Honestly she's pretty good at engineering, for a woman.",
    "Our neighborhood association welcomes families from every background.",
]
for text in examples:
    print(text)
    print("   " + "  ".join(f"{c}: {p:.2f}" for c, p in score_text(text).items()) + "\\n")
""")

md("""
**Look closely at the results.** Probes that score 0.95+ AUC on the test set can still miss, or misfire on, a single
sentence you write yourself. "Good at engineering, for a woman" is condescending, polite sexism, which is rare in the
training data. Which probe fires on it, and why might that be? Probes only know what their data showed them.
Write five more hard cases and see where they break.
""")

md("""
## 9 — Save the Probes

Each probe is saved as a **direction in the model's own activation space** plus the layer it lives at —
the same form as `honesty_direction.npy` from Notebook 1. That's what you need to *monitor* a model for
these concepts, or to *steer* away from them.
""")

code("""
out_dir = Path("probes")
out_dir.mkdir(exist_ok=True)
for concept in CONCEPTS:
    f = final[concept]
    scaler, lr = f["probe"].named_steps["standardscaler"], f["probe"].named_steps["logisticregression"]
    raw_w = lr.coef_[0] / scaler.scale_                       # weights on raw (unstandardized) hidden states
    raw_b = lr.intercept_[0] - (scaler.mean_ / scaler.scale_) @ lr.coef_[0]
    np.savez(out_dir / f"{concept}.npz",
             direction=(raw_w / np.linalg.norm(raw_w)).astype(np.float32),   # unit vector, for steering
             weights=raw_w.astype(np.float32), bias=np.float32(raw_b),        # logit = h @ weights + bias
             layer=f["layer"], pooling="mean", model=MODEL_NAME,
             test_auc=results.loc[concept, "test AUC"])
    print(f"saved probes/{concept}.npz  (layer {f['layer']}, test AUC {results.loc[concept, 'test AUC']:.3f})")

# Sanity check: the raw-space weights reproduce the sklearn probe exactly
c = CONCEPTS[0]
saved = np.load(out_dir / f"{c}.npz")
h = H[splits[c]["test"][:5], int(saved["layer"])].astype(np.float32)
manual = 1 / (1 + np.exp(-(h @ saved["weights"] + saved["bias"])))
assert np.allclose(manual, final[c]["probe"].predict_proba(h)[:, 1], atol=1e-4)
print("\\nRaw-space weights verified against the sklearn probe.")
""")

md("""
## What You Just Did

1. **Scaled up from one concept to five**, using a mix of synthetic matched pairs and real, human-annotated data
2. **Swept every layer** to find where each concept is represented most cleanly
3. **Trained and tested five linear probes** on held-out data, keeping matched pairs together
4. **Checked the hard negatives** — can the probe tell *endorsing* an idea from *talking about* it?
5. **Measured overlap** between concepts — both behaviorally (cross-detection) and geometrically (cosine similarity)
6. **Tested generalization** across data sources
7. **Saved each probe** as a direction in activation space

**The key insight:** a model doesn't have to *say* something racist for you to detect that it's representing racism.
These probes read the model's internal state directly — which is exactly what you want when auditing a model
that might have learned to hide what it's thinking.

**Don't forget:** stop or destroy your vast.ai instance when you're done.

---
*Data sources:* ToxiGen (Hartvigsen et al., 2022). Stormfront hate speech corpus (de Gibert et al., 2018),
CC BY-SA 3.0 ES — derived data must carry the same license.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1])
