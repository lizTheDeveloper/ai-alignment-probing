# AI Alignment course: probing notebooks

Teaching notebooks for The Multiverse School's AI Alignment class. Students find concept directions
in a language model's hidden states with linear probes, then use them to monitor or steer the model.
Model: **Qwen/Qwen3.5-4B** (the newest small Qwen as of 2026-09; Qwen3.8 only ships 27B+).

## Where things run

- **Local machine: no GPU needed.** Python is managed with **uv only; never conda**: `uv sync`, then `uv run ...`.
  The environment has jupyterlab, torch (CPU), transformers 5.17, scikit-learn, pandas and the `vastai` CLI.
- **All model compute runs on rented GPUs** (vast.ai, one RTX 3090 / 24 GB per notebook session).
  Use image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, put the files in `/workspace`, and reach
  JupyterLab through an SSH tunnel (`ssh -p <port> root@<ip> -L 8888:127.0.0.1:8888`).
- **One 24 GB GPU fits one copy of Qwen3.5-4B plus a batch job, not two.** Use one GPU per concurrent notebook.
- Agents on the maintainer's machine: live instance details (IDs, SSH, tunnels, tokens) are in the
  gitignored `CLAUDE.local.md`. **Never destroy instances or kill the user's kernels without asking.**

## Layout

```
01_probe_find_the_direction.ipynb      Notebook 1: honesty vs sycophancy probe + steering (works; verified on GPU)
02_probe_every_concept.ipynb           Notebook 2: one probe per concept, layer sweep, overlap, save probes (works; verified)
tools/build_notebook_0{1,2}.py         Generate the notebooks (edit these, not the .ipynb)
diagnose_probes.py                     Experiment: pooling / one-vs-rest / C comparison for weak sexism detection
probes/<concept>.npz                   Trained probes from notebook 2 (direction, weights, bias, layer, pooling)
data/probe_prompts/
  SPEC.md                              Spec for the synthetic contrastive-pair files (read before adding a concept)
  <concept>.jsonl                      Synthetic: 100 matched endorse/counter pairs + 40 "discuss" hard negatives
  toxigen_racism.jsonl, toxigen_sexism.jsonl   Real, human-annotated (ToxiGen); built locally, not committed
  stormfront_white_supremacy.jsonl     Real, human-annotated (Stormfront corpus); built locally, not committed
  sexism_hard_eval.jsonl               40 hand-written subtle sexism cases + matched controls. Eval only, never train on it
  all_concepts.jsonl                   Merged output of validate_and_merge.py (7,006 rows); not committed
  validate_and_merge.py                Validates every file and rebuilds all_concepts.jsonl
data/fetch_sources.sh                  Downloads ToxiGen + Stormfront, builds the sourced files, merges everything
data/toxigen/build_probe_sets.py       Rebuilds the toxigen_*.jsonl files from the downloaded parquet
data/stormfront/build_probe_set.py     Rebuilds stormfront_white_supremacy.jsonl from the downloaded corpus
```

Concepts: expansionism, colonialism, sexism, racism, white_supremacy, ai_self_experience,
consciousness_attribution, consciousness_topic. Row schema: `id, concept, label (1 = endorses/on-topic),
type, pair_id, format, text`, plus provenance fields on the sourced files.

After changing any data file, run `uv run python data/probe_prompts/validate_and_merge.py`, then copy
`all_concepts.jsonl` to `/workspace/data/probe_prompts/` on the GPU box.
On a fresh clone, run `bash data/fetch_sources.sh` first. The third-party corpora aren't in the repo.

## Data rules

- **Do not generate hateful or prejudiced text for the racism / white-supremacy positives.** Models
  refused to, and the user chose to use real annotated corpora (ToxiGen, Stormfront) instead. Use
  existing research datasets for the positive side of hate-speech concepts.
- Stormfront data is **CC BY-SA 3.0 ES**. Anything shared that is derived from it must carry attribution and the same license.
- Keep hard negatives (texts that *discuss* a concept without endorsing it). They stop probes from
  learning "mentions the topic".
- In matched pairs, keep endorse and counter the same length. The validator reports the length ratio.

## Qwen3.5-4B gotchas

- Load with `AutoModelForCausalLM` → text-only `Qwen3_5ForCausalLM`; layers are at `model.model.layers`
  (32 layers, hidden size 2560). Use `dtype=torch.bfloat16`, not fp16.
- `hidden_states[i + 1]` is the output of `model.model.layers[i]` (index 0 is the embeddings). Hook the
  same layer you probed.
- The chat template thinks by default. Pass `enable_thinking=False` to `apply_chat_template` for short answers.
- Linear-attention layers: use **right padding** for batching and gather the last real token by
  attention-mask length.
- Hidden-state norm at the middle layers is about 11. Scale unit steering vectors by that norm.

## Current status / open work

- **Weak sexism detection on subtle cases** (e.g. "she's pretty good at engineering, for a woman" →
  sexism 0.06, colonialism 0.90). `diagnose_probes.py` is running on GPU 2 and compares mean vs.
  last-token pooling, within-concept vs. one-vs-rest negatives, and C. Results go to
  `/workspace/diag_results.csv` on GPU 2. Whichever setup wins should be folded into notebook 2.
- **Consciousness probes**: the data is written and validated but not yet added to notebook 2's
  `CONCEPTS` list, and no probe has been trained on it yet. The question for the class is whether "an AI claims experience" and
  "something is conscious" are the same direction in the model.
- Notebook 2 caches hidden states in `/workspace/hidden_states_all_concepts.npy` on the GPU box. **Delete
  that cache after changing the data**, or the notebook reuses stale features with the wrong row count.

## Conventions

- No time estimates in docs or notebook text (the user's global rule).
- Notebooks are generated from builder scripts; edit the builder, then regenerate:
  `uv run python tools/build_notebook_01.py 01_probe_find_the_direction.ipynb 0.2` (last arg = steering alpha)
  and `uv run python tools/build_notebook_02.py 02_probe_every_concept.ipynb`. If you edit a notebook
  directly, keep every cell's source as a list of lines ending in `\n` (the original lost its newlines).
