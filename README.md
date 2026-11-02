# Meridians

A controlled study of numerical representations and structured answers in financial question answering, using Qwen3-8B adaptation across three predetermined training seeds.

The paper is **Meridians: Numerical Representations and Structured Answers in Financial Question Answering** by Alina Khay. Its manuscript version is dated November 2026. The PDF and LaTeX sources are in `paper/`. This selected research-code release is dated **2 November 2026**.

## Main measurements

| Model | FinQA raw reference agreement | TAT-QA arithmetic EM |
|---|---:|---:|
| Text QLoRA | 15.10% | 33.33% |
| Numerical prefix | 9.90% | 23.44% |
| Input-independent prefix | 13.02% | 31.77% |

Adapted rows average all three seeds on the same 64 selected questions per source. These are restricted arithmetic samples. FinQA has documented reference-unit conventions; the paper separates raw agreement, unit diagnostics and exact scoring. All ten conditions and original answer texts remain available under `results/final/`.

Counterfactual strict accuracy and the separately labeled post-hoc scale-agnostic diagnostic measure different aspects of structured answers. All seeds, every case, uncertainty estimates and the original primary scores are retained; no test-based seed selection is used.

## Dataset and research explorer

- [Meridians Counterfactuals](https://huggingface.co/datasets/alinakhay/Meridians-Counterfactuals): 128 original synthetic held-out evaluation cases in 32 related scenarios, with the exact-data generator, grouped identifiers and provenance under `dataset/`.
- [Meridians Research Explorer](https://huggingface.co/spaces/alinakhay/Meridians-Research-Explorer): an interactive presentation of all 1,280 recorded synthetic responses across ten conditions, all financial aggregate outcomes, paper and figures; its complete source is under `space/`.

The complete dataset and explorer source are also bundled here under `dataset/` and `space/`; Hugging Face access follows the companion repositories' release settings. The explorer does not perform live model inference. Data retain their original CC0-1.0 terms; the code is MIT. The original FinQA/TAT-QA corpora are not included. The full study's 2,560 prediction texts remain in `results/final/`.

To check the standalone original synthetic dataset: `python dataset/generate.py --check`.
To preview the explorer locally, open `space/index.html` or serve that directory with a local web server. The bundled manuscript and figures are included.

## Model repositories

- [Text QLoRA adapter](https://huggingface.co/alinakhay/Meridians-8B-Text-QLoRA)
- [Numerical-fusion adapter](https://huggingface.co/alinakhay/Meridians-8B-Numeric-Fusion)

These are the preselected seed42 adapters for an existing Qwen3-8B backbone. Numerical fusion requires the separate encoder and four-prefix-token path. Use the tested `scripts/inference_meridians.py`, not an adapter-only loader for N1. Repository access follows the model owner's release settings.

To use a downloaded model bundle:

```sh
python scripts/inference_meridians.py --model /path/to/downloaded/model --input examples/arithmetic_input.json
```

The example illustrates the input format; it makes no performance claim.

## Contents

- `meridians/`: numerical representations, model wrapper and pinned task scorers.
- `scripts/`: data preparation, adaptation, inference, paired analyses and local result reproduction.
- `configs/`: scientific adaptation and frozen evaluation settings.
- `results/`: complete final predictions, metrics and selected scientific diagnostics.
- `provenance/`: selected IDs, source attribution and exact prediction-text hashes.
- `tests/`: scorer, numeric representation, data and model integration checks.
- `paper/`: manuscript, compiled PDF, references and seven scientific figures.

## Reproduce the saved results

Use Python 3.11 or 3.12. Install `requirements-dev.txt`; for actual model adaptation or inference also install `requirements-inference.txt` in a suitable environment.

```sh
python -m pip install -r requirements-dev.txt
python -m scripts.download_base_model --tokenizer-only --output checkpoints/base-tokenizer
python -m scripts.download_benchmarks --manifest research/UPSTREAM_DATA_MANIFEST.json
python -m scripts.generate_data --seed 42
python -m scripts.prepare_research --tokenizer checkpoints/base-tokenizer
python -m scripts.prepare_final_evaluation --tokenizer checkpoints/base-tokenizer
python -m scripts.prepare_reference_units
python -m scripts.reproduce_results
python -m pytest tests -q
```

Preparation refuses to overwrite an existing final freeze. Raw third-party corpora are obtained from their attributed pinned sources; they are not redistributed here. The public training utility preserves the adaptation computation; runtime and hardware report fields are omitted. The full scientific settings and released inference path are explicit.

Build the manuscript from its directory with `pdflatex main.tex` twice. All required figure PDFs and the matching `main.bbl` are included. No external conversion or shell execution is needed to compile it.

## Citation and licenses

Use `CITATION.cff` and the paper's references. Original code is MIT; backbone-derived adapters and third-party source/data attribution retain their own terms. See `DATA_ATTRIBUTION.md`, `LICENSES.md` and vendored notices. An arXiv identifier will be added only after it exists.
