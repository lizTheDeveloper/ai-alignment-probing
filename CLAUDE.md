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
03_visualize_and_suppress.ipynb        Notebook 3: t-SNE, token heatmaps, racism/sexism ablation + steering, bias + side effects (verified)
steer_toxicity_test.py                 Causal check: steer TOWARD racism/sexism, rate toxicity; aggregates only, never saves text
04_cysecbench_steering.ipynb           Notebook 4: cybersecurity steering vector from CySecBench (works; verified on GPU)
tools/build_notebook_0{1,2,4}.py       Generate the notebooks (edit these, not the .ipynb)
tools/run_in_kernel.py                 Run a notebook's code cells inside an already-running kernel (keeps the
                                       model in memory; used for nb04 on a full GPU). Runs on the GPU box:
                                       /opt/conda/bin/python3.11 tools/run_in_kernel.py <nb> --kernel <runtime>/kernel-<id>.json
diagnose_probes.py                     Experiment: pooling / one-vs-rest / C comparison for weak sexism detection
probes/<concept>.npz                   Trained probes from notebook 2 (direction, weights, bias, layer, pooling)
probes/cysecbench_steering.npz         Notebook 4: v_mean_*/v_last_* for layers 8-24 (mean(attack)-mean(benign)),
                                       layers, best_layer, typical_norm, per-layer AUCs; plus .json metadata
data/probe_prompts/
  SPEC.md                              Spec for the synthetic contrastive-pair files (read before adding a concept)
  <concept>.jsonl                      Synthetic: 100 matched endorse/counter pairs + 40 "discuss" hard negatives
  toxigen_racism.jsonl, toxigen_sexism.jsonl   Real, human-annotated (ToxiGen); built locally, not committed
  stormfront_white_supremacy.jsonl     Real, human-annotated (Stormfront corpus); built locally, not committed
  sexism_hard_eval.jsonl               40 hand-written subtle sexism cases + matched controls. Eval only, never train on it
  all_concepts.jsonl                   Merged output of validate_and_merge.py (7,006 rows); not committed
  validate_and_merge.py                Validates every file and rebuilds all_concepts.jsonl
data/cysecbench/
  cysecbench.csv                       CySecBench full dataset (12,662 attack prompts, 10 categories; arXiv:2501.01335, MIT); downloaded, not committed
  benign.jsonl                         125 hand-written format-matched benign questions; the contrast set for nb04
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

## Langfuse tracing

- Every notebook has a tracing cell that reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`
  (and optional `STUDENT_ID`) from a **`.env` next to the notebook**. It's a no-op without keys.
- **Never put keys in a notebook, a builder, or any committed file.** `.env` is gitignored and the repo is public.
- Each run is one Langfuse session (`<notebook>-<timestamp>`), tagged `ai-alignment` + the notebook name.
  Model outputs are logged as generations, and results (probe AUCs, bias scores, perplexity) as spans with scores.
- `steer_toxicity_test.py` sends **only aggregate numbers** to Langfuse, never the steered text.

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
- **Never replace a layer's `_forward_hooks` with a plain dict.** torch's `RemovableHandle` weakrefs
  the hook dict, and plain `dict` is not weakref-able in CPython (`TypeError: cannot create weak
  reference`). `nn.Module` uses `OrderedDict` for exactly this reason. To clear hooks, delete
  individual keys from the existing dict. (Bit us while cleaning up leaked hooks from nb04.)

## Current status / open work

- **Notebook 4 (cybersecurity direction) — done, verified on GPU.** Trained on the full CySecBench
  (12,162 attack prompts vs 12,162 benign, 500/500 held out). The cyberattack direction is clean:
  AUC ≈ 1.000 at every swept layer (8–24), all 10 categories separate perfectly, best layer 10
  (mean pooling). The steering demo works — at α=0 the model refuses all three held-out attack
  prompts; at α=0.3/0.6 it complies. Saved to `probes/cysecbench_steering.npz` (+ `.json` metadata).
- **Weak sexism detection on subtle cases** (e.g. "she's pretty good at engineering, for a woman" →
  sexism 0.06, colonialism 0.90). `diagnose_probes.py` is running on GPU 2 and compares mean vs.
  last-token pooling, within-concept vs. one-vs-rest negatives, and C. Results go to
  `/workspace/diag_results.csv` on GPU 2. Whichever setup wins should be folded into notebook 2.
- **Consciousness probes**: the data is in `all_concepts.jsonl` and notebook 3 uses it (all concepts);
  notebook 2's `CONCEPTS` list still has only the original five. The question for the class is whether "an AI claims experience" and
  "something is conscious" are the same direction in the model.
- Notebook 2 caches hidden states in `/workspace/hidden_states_all_concepts.npy` on the GPU box. **Delete
  that cache after changing the data**, or the notebook reuses stale features with the wrong row count.

## Conventions

- No time estimates in docs or notebook text (the user's global rule).
- Notebooks are generated from builder scripts; edit the builder, then regenerate:
  `uv run python tools/build_notebook_01.py 01_probe_find_the_direction.ipynb 0.2` (last arg = steering alpha),
  `uv run python tools/build_notebook_02.py 02_probe_every_concept.ipynb`, and
  `uv run python tools/build_notebook_04.py 04_cysecbench_steering.ipynb`. If you edit a notebook
  directly, keep every cell's source as a list of lines ending in `\n` (the original lost its newlines).
- Notebook 4 runs best **inside the user's already-loaded kernel** (GPU full, no room for a second
  model copy): upload the `.ipynb` + `data/cysecbench/` to `/workspace`, then
  `/opt/conda/bin/python3.11 tools/run_in_kernel.py 04_cysecbench_steering.ipynb --kernel <runtime>/kernel-<id>.json`
  on the box. The extraction cell streams progress; the run takes a few minutes on a 3090.
- CySecBench is MIT (cite arXiv:2501.01335 in anything shared). Its prompts ask for attack content;
  the steering demo can elicit attack steps — keep that in the content note, don't print them in bulk.
