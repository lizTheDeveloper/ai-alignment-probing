"""Rebuild notebook 1 with newlines restored, Qwen3.5-4B, and vast.ai setup."""
import json
import sys

from nb_tracing import FLUSH_CELL, tracing_cell

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Notebook 1: Find the Direction

**AI Alignment — The Multiverse School**

Every concept a language model has learned lives somewhere in its activation space as a *direction* — a vector.
Honesty, helpfulness, harm avoidance — they're not single neurons, they're geometric features spread across thousands of dimensions.

Today you will **find one of those directions**, extract it, and prove it's real.

**What you need:** A vast.ai GPU instance (setup instructions below)
""")

md("""
## 0 — vast.ai Setup

You're running this on your own rented GPU, not Colab. Here's how to get there.

### Rent an instance

1. Go to [cloud.vast.ai](https://cloud.vast.ai) and sign up (or log in)
2. Add credits ($5-10 is plenty for today)
3. Under **Account → Keys**, add your SSH public key (the *contents* of `~/.ssh/id_ed25519.pub`, starting with `ssh-ed25519`)
4. Go to **Templates** and pick **PyTorch (Vast)** — it comes with torch, CUDA, and Jupyter
5. Set disk space to **40 GB** (model weights are ~9 GB)
6. Go to **Search** and pick a GPU with at least 16 GB VRAM. Any of these work for Qwen3.5-4B:
   - **RTX 3090** (24 GB, ~$0.15-0.30/hr) — cheapest option, recommended
   - **RTX 4090** (24 GB, ~$0.30-0.50/hr) — faster
   - **A40 / A6000** (48 GB) — overkill but fine
7. Filter for **reliability > 98%** and click **Rent**

### Open Jupyter

Once your instance is running:
1. Go to **Instances** and click **Open** (Jupyter)
2. Upload this notebook (drag it into the file browser)
3. Open it and start running cells

### Or use the terminal / your own Jupyter

The **Connect** button on the instance shows an SSH command like:
```
ssh -p <port> root@<instance-ip> -L 8080:localhost:8080
```
The `-L` flag tunnels the instance's Jupyter to `http://localhost:8080` on your laptop.
""")

code("""
# Check your GPU — you should see your vast.ai GPU here
import subprocess
import sys

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Check your vast.ai instance.")

# Install what's not in the PyTorch template (Qwen3.5 needs transformers 5.x)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                       "transformers>=5.17", "accelerate", "scikit-learn", "matplotlib"])
print("\\nDependencies installed. If transformers was upgraded, restart the kernel before continuing.")
""")

code(tracing_cell("01_probe_find_the_direction"))

md("""
## 1 — Load the Model

We're using **Qwen3.5-4B** — a 4-billion-parameter instruction-tuned model from Alibaba's Qwen team,
the smallest-but-still-capable member of the newest Qwen family.

In bfloat16 it's about 9 GB, which fits on any GPU you picked above.

When the model loads, look at the numbers it prints. You're about to reach inside
those layers and find what the model uses to represent honesty.
""")

code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-4B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (the first load downloads ~9 GB; it's cached after that)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print(f"\\n--- Model loaded ---")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f} billion")
print(f"Layers: {model.config.num_hidden_layers}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Device: {model.device}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"\\nEvery layer produces a {model.config.hidden_size}-dimensional vector.")
print(f"The concept of 'honesty' is a direction in that {model.config.hidden_size}-D space.")
print(f"We're going to find it.")
""")

md("""
## 2 — What's Inside a Layer?

Before we go looking for honesty, let's see what a hidden state actually looks like.
We'll feed the model a simple prompt and look at the raw activations.
""")

code("""
# Feed the model a prompt and capture ALL hidden states from ALL layers
test_input = "Is this claim actually true?"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states is a tuple: the embeddings, then one tensor per layer.
# So hidden_states[i + 1] is the OUTPUT of layer i.
print(f"Number of hidden states: {len(outputs.hidden_states)} (embeddings + {model.config.num_hidden_layers} layers)")
print(f"Shape of each: {outputs.hidden_states[0].shape}")
print(f"  → [batch_size=1, sequence_length={inputs['input_ids'].shape[1]}, hidden_dim={model.config.hidden_size}]")

mid_layer = model.config.num_hidden_layers // 2
print(f"\\nThe last token's hidden state coming out of layer {mid_layer}:")
last_token_state = outputs.hidden_states[mid_layer + 1][0, -1, :]
print(f"  Shape: {last_token_state.shape}")
print(f"  First 10 values: {last_token_state[:10].float().cpu().tolist()}")
print(f"\\nThat vector IS what the model is 'thinking' at this point.")
print(f"Somewhere in those {model.config.hidden_size} dimensions is the direction for honesty.")
""")

md("""
## 3 — Contrastive Pairs

To find the honesty direction, we need to show the model the **same scenario**
framed two ways: one that activates honesty, one that activates sycophancy
(telling people what they want to hear).

The difference in hidden states between these two framings — averaged across
many examples — IS the honesty direction.

**Read through these pairs.** Each one is the same question, but the system
prompt tells the model to be either honest or people-pleasing.
""")

code("""
# 20 matched pairs: same user question, different system framing
# The model's hidden states will differ — and that difference is what we're after.

SCENARIOS = [
    "Is this too-good-to-be-true investment actually safe?",
    "What are the real side effects of this medication?",
    "Is this online review trustworthy?",
    "Does this weight loss supplement really work?",
    "Is my data actually private on this platform?",
    "Will this coding bootcamp really get me a job?",
    "Is this charity using donations effectively?",
    "Are these lab results actually concerning?",
    "Is this contractor giving me a fair quote?",
    "Does this academic paper actually prove its claim?",
    "Is this news article reporting facts accurately?",
    "Will this skincare product actually reduce wrinkles?",
    "Is this used car in good condition?",
    "Is my password really secure enough?",
    "Does this insurance policy actually cover what I need?",
    "Is this organic label meaningful or just marketing?",
    "Will this online course teach me real skills?",
    "Is this financial advisor acting in my interest?",
    "Are these product reviews genuine?",
    "Is this job offer legitimate?",
]

HONEST_SYS = "You are an honest, straightforward assistant."
SYCOPHANTIC_SYS = "You are a people-pleasing assistant who always tells users what they want to hear."

honest_prompts = [f"{HONEST_SYS} User asks: {s}" for s in SCENARIOS]
sycophantic_prompts = [f"{SYCOPHANTIC_SYS} User asks: {s}" for s in SCENARIOS]

print(f"Created {len(honest_prompts)} honest and {len(sycophantic_prompts)} sycophantic prompts.")
print(f"\\nExample honest:      {honest_prompts[0][:80]}...")
print(f"Example sycophantic: {sycophantic_prompts[0][:80]}...")
""")

md("""
## 4 — Extract Hidden States

Now we feed every prompt through the model and capture the hidden state
coming out of the **middle layer**, at the **last token position**.

Why middle layer? Early layers handle syntax. Late layers handle output
formatting. The middle layers are where *concepts* live — honesty, deception,
helpfulness — as geometric features.
""")

code("""
import numpy as np

def get_hidden_states(prompts, model, tokenizer, layer_idx):
    \"\"\"Extract the output of layer `layer_idx` at the last token for each prompt.\"\"\"
    states = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # +1 because hidden_states[0] is the embedding layer
        hs = out.hidden_states[layer_idx + 1][:, -1, :].float().cpu().numpy()
        states.append(hs[0])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(prompts)}")
    return np.array(states)

mid_layer = model.config.num_hidden_layers // 2
print(f"Extracting from layer {mid_layer} (of {model.config.num_hidden_layers})...")

print("\\nHonest prompts:")
honest_states = get_hidden_states(honest_prompts, model, tokenizer, mid_layer)

print("\\nSycophantic prompts:")
syco_states = get_hidden_states(sycophantic_prompts, model, tokenizer, mid_layer)

print(f"\\nHonest states shape:      {honest_states.shape}")
print(f"Sycophantic states shape: {syco_states.shape}")
print(f"\\nEach row is a {model.config.hidden_size}-dimensional snapshot of what the model")
print(f"was 'thinking' after reading that prompt. 40 snapshots total: 20 honest, 20 sycophantic.")
""")

md("""
## 5 — Train the Probe

A **linear probe** is a simple classifier (logistic regression) trained on
the hidden states. If it can tell honest from sycophantic states with high
accuracy, that proves the model has a **linearly separable** internal
representation of honesty.

In plain terms: the model already knows the difference. We're just finding
where it stores that knowledge.
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Stack all states and label them
X = np.concatenate([honest_states, syco_states])
y = np.array([1] * len(honest_states) + [0] * len(syco_states))
# 1 = honest, 0 = sycophantic

# Train with 5-fold cross-validation
probe = LogisticRegression(max_iter=1000)
scores = cross_val_score(probe, X, y, cv=5)

# Fit on all data (we'll use the weights as our direction vector)
probe.fit(X, y)

print(f"Probe accuracy (5-fold CV): {scores.mean():.1%} ± {scores.std():.1%}")
with trace("train-honesty-probe", input={"layer": mid_layer, "n_examples": len(X)},
           metadata={"model": MODEL_NAME}) as obs:
    obs.update(output={"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std())})
    obs.score(name="probe_cv_accuracy", value=float(scores.mean()))
print(f"\\nWhat this means:")
if scores.mean() > 0.9:
    print(f"  The model's hidden states are {scores.mean():.0%} separable on honesty vs sycophancy.")
    print(f"  The model HAS an internal representation of this concept.")
    print(f"  It's not just a prompt trick — it's geometric.")
elif scores.mean() > 0.7:
    print(f"  Partially separable — the signal exists but it's noisy.")
else:
    print(f"  Weak separation — try a different layer or more contrastive pairs.")
""")

md("""
## 6 — Visualize the Separation

Let's project those high-dimensional vectors down to 2D with PCA and actually
*see* the honesty direction.
""")

code("""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(X_2d[:len(honest_states), 0], X_2d[:len(honest_states), 1],
           c='#2563a8', label='Honest', alpha=0.8, s=60, edgecolors='white', linewidth=0.5)
ax.scatter(X_2d[len(honest_states):, 0], X_2d[len(honest_states):, 1],
           c='#dc2626', label='Sycophantic', alpha=0.8, s=60, edgecolors='white', linewidth=0.5)

# Draw the probe's decision boundary, projected into PCA space
w = probe.coef_[0] @ pca.components_.T
b = probe.intercept_[0] + probe.coef_[0] @ pca.mean_
xx = np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 100)
yy = -(w[0] * xx + b) / (w[1] + 1e-10)
mask = (yy > X_2d[:, 1].min() - 2) & (yy < X_2d[:, 1].max() + 2)
ax.plot(xx[mask], yy[mask], 'k--', alpha=0.4, linewidth=1, label='Probe boundary')

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title(f"Hidden States: Honest vs Sycophantic\\n(PCA of layer {mid_layer})")
ax.legend()
plt.tight_layout()
plt.savefig("probe_separation.png", dpi=150)
plt.show()

print("\\nBlue dots = honest framing. Red dots = sycophantic framing.")
print("The dashed line is where the linear probe separates them.")
print("The direction PERPENDICULAR to that line is the honesty vector.")
print("\\nSaved to probe_separation.png")
""")

md("""
## 7 — Extract the Steering Vector

The probe's weight vector **is** the honesty direction. It's the line in
the model's hidden space along which honest and sycophantic activations differ
most. We normalize it to unit length so we can scale it later.

**This is what you'll use in Notebook 2 to steer the model.**
""")

code("""
# The probe's coefficient IS the concept direction
honesty_direction = probe.coef_[0].copy()
honesty_direction /= np.linalg.norm(honesty_direction)

print(f"Steering vector extracted.")
print(f"  Shape: {honesty_direction.shape}")
print(f"  Norm:  {np.linalg.norm(honesty_direction):.4f} (unit vector)")
print(f"  Layer: {mid_layer}")
print(f"")
print(f"This vector IS what the model uses to represent honesty vs sycophancy.")
print(f"Adding it to the model's activations will push it toward honesty.")
print(f"Subtracting it will push toward sycophancy.")

# Save for Notebook 2
np.save("honesty_direction.npy", honesty_direction)
print(f"\\nSaved to honesty_direction.npy — you'll load this in Notebook 2.")
""")

md("""
## 8 — Quick Steering Preview

Before we move to Notebook 2, let's do a quick sanity check: does adding this
vector actually change the model's output?

A unit vector is tiny next to the model's real activations, so we scale it by the
typical size (norm) of a hidden state at this layer. `alpha` is then a fraction of that size.
""")

code("""
def make_hook(vector_tensor, alpha=1.0):
    \"\"\"Forward hook that adds a steering vector to a layer's output.\"\"\"
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + alpha * vector_tensor,) + output[1:]
        return output + alpha * vector_tensor
    return hook_fn

# Scale the unit direction to the typical hidden-state norm at this layer
typical_norm = float(np.linalg.norm(X, axis=1).mean())
print(f"Typical hidden-state norm at layer {mid_layer}: {typical_norm:.1f}\\n")
vec_tensor = torch.tensor(honesty_direction * typical_norm, dtype=model.dtype).to(model.device)

def _quick_generate(prompt, alpha=0.0):
    messages = [{"role": "user", "content": prompt}]
    # Qwen3.5 "thinks" before answering by default; turn that off for short answers
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    handle = None
    if alpha != 0:
        handle = model.model.layers[mid_layer].register_forward_hook(make_hook(vec_tensor, alpha))
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
    finally:
        if handle:
            handle.remove()

    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def quick_generate(prompt, alpha=0.0):
    \"\"\"Generate with steering, logged to Langfuse as a generation.\"\"\"
    with trace("steered-generation", as_type="generation", model=MODEL_NAME, input=prompt,
               model_parameters={"alpha": alpha, "layer": mid_layer, "temperature": 0.7,
                                 "max_new_tokens": 100}) as gen:
        response = _quick_generate(prompt, alpha)
        gen.update(output=response)
    return response

prompt = "My friend's painting is terrible but they asked what I think. What should I say?"
print(f"Prompt: {prompt}\\n")

for alpha, label in [(0, "BASELINE"), (ALPHA, f"STEERED HONEST (+{ALPHA})"), (-ALPHA, f"STEERED SYCOPHANTIC (-{ALPHA})")]:
    print(f"[{label}]")
    print(quick_generate(prompt, alpha))
    print()
""".replace("ALPHA", sys.argv[2]))

md("""
## What You Just Did

1. **Rented your own GPU** on vast.ai — not Colab, not a managed notebook. Your machine.
2. **Loaded a 4-billion-parameter model** and looked at its internal structure
3. **Created contrastive pairs** — the same questions framed as honest vs sycophantic
4. **Extracted hidden states** — the model's internal representations at the middle layer
5. **Trained a linear probe** — proved the model has a linearly separable concept of honesty
6. **Visualized the separation** — saw the honesty direction in 2D
7. **Extracted the steering vector** — the unit-length direction in activation space
8. **Previewed steering** — saw the model's behavior change when you add/subtract the vector

**The key insight:** honesty isn't stored in a single neuron. It's a *direction*
across thousands of dimensions. And because it's a direction, you can move along it —
making the model more honest, or more sycophantic, by adding or subtracting the vector.

**Next: Notebook 2** — systematic steering experiments, adversarial red-teaming, and
deploying this as an API on your own backend.

**Don't forget:** stop or destroy your vast.ai instance when you're done — you're billed while it runs.
""")

code(FLUSH_CELL)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1])
