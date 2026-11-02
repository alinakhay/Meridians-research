# Data and source attribution

Original code: MIT, Alina Khay, 2026. Qwen/Qwen3-8B backbone-derived adapters retain Apache-2.0 terms and are distributed separately from this code repository.

FinQA: Chen et al., EMNLP 2021. [Paper](https://aclanthology.org/2021.emnlp-main.300/), [official repository](https://github.com/czyssrs/FinQA), [IBM Research dataset card](https://huggingface.co/datasets/ibm-research/finqa). The upstream repository provides an MIT license; the dataset mirror identifies CC BY 4.0 data attribution. Original source revisions, hashes and attribution are retained in research/UPSTREAM_DATA_MANIFEST.json. FinQA's official executor is vendored with its source manifest and MIT notice under third_party/finqa/.

TAT-QA: Zhu et al., ACL-IJCNLP 2021. [Paper](https://aclanthology.org/2021.acl-long.254/), [official repository](https://github.com/NExTplusplus/TAT-QA). Dataset attribution is CC BY 4.0; upstream code MIT notices and exact evaluator source hashes are retained under meridians/evaluation/tatqa/.

This release includes selected question IDs, derived metric/reference diagnostics and model outputs. Raw upstream corpora are not included; download the pinned public sources with the preparation scripts. No claim is made that synthetic-data terms cover third-party data. The original synthetic generator declares CC0-1.0 for its generated examples; the new quartet suite is derived from those generated scenarios.

The selection manifests and reference-unit sidecar describe transformations for this study; the complete upstream benchmarks remain distinct. FinQA unit-canonical reference agreement is a secondary local diagnostic, not an official metric.

## Meridians Counterfactuals release

`dataset/` contains only the 128 authored synthetic evaluation examples in 32 related scenarios and their generation seeds, under the original CC0-1.0 data dedication. Its generator uses MIT (see nested CODE_LICENSE). `space/` embeds those same cases and recorded experimental answers; it contains no original FinQA/TAT-QA question text. Data and presentation-code licenses are separate; neither license reassigns rights in upstream datasets or model weights.
