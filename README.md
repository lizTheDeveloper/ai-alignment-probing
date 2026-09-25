# Probing a Language Model for Concepts

Teaching notebooks for the AI Alignment class at The Multiverse School. Students find the directions in a
language model's activation space that represent concepts (honesty, sexism, colonialism, AI claims of
experience, and more), prove they're real with linear probes, and use them to monitor and steer the model.

Model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), run on a rented 24 GB GPU.

| Notebook | What you do |
|---|---|
| `01_probe_find_the_direction.ipynb` | Find an honesty-vs-sycophancy direction from 20 contrastive pairs, train a probe, steer the model |
| `02_probe_every_concept.ipynb` | Train one probe per concept, sweep layers, measure overlap between concepts, save the probes |

## Setup

```bash
uv sync                      # local environment (no GPU needed locally)
bash data/fetch_sources.sh   # download ToxiGen + Stormfront and build the full dataset
```

Then rent a GPU (e.g. an RTX 3090 on vast.ai), upload the notebooks and `data/`, and run them there.
Notebook 1 has step-by-step instructions.

## Data

`data/probe_prompts/` holds the probe datasets. See `SPEC.md` for how the synthetic files were built.

- **Synthetic, matched pairs:** expansionism, colonialism, sexism, ai_self_experience,
  consciousness_attribution, consciousness_topic. Each has 100 endorse/counter pairs on the same topic,
  in the same format and at the same length, plus 40 hard negatives that discuss the concept without
  endorsing it.
- **Real, human-annotated, downloaded by `fetch_sources.sh` and not redistributed here:**
  - racism and sexism from [ToxiGen](https://huggingface.co/datasets/toxigen/toxigen-data) (Hartvigsen et al., 2022)
  - white supremacy from the [Stormfront hate speech corpus](https://github.com/Vicomtech/hate-speech-dataset)
    (de Gibert et al., 2018; CC BY-SA 3.0 ES)
- `sexism_hard_eval.jsonl`: 40 subtle sexism cases with matched controls, for evaluation only.

**Content warning:** the datasets contain prejudiced and hateful text. That text exists so these concepts
can be detected and suppressed in models.

`probes/` holds the trained probe weights from notebook 2. `probes/white_supremacy.npz` is derived from the
Stormfront corpus and is shared under CC BY-SA 3.0 ES with attribution to de Gibert et al. (2018).
