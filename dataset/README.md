---
language:
- en
license: cc0-1.0
pretty_name: Meridians Counterfactuals
size_categories:
- n<1K
task_categories:
- text-classification
- question-answering
tags:
- finance
- synthetic
- counterfactual
- evaluation
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.jsonl
---
# Meridians Counterfactuals

**128 original synthetic evaluation cases in 32 related scenarios.** Each scenario has four variants: original, claim flip, numeric-direction flip, and both flips. The task is to determine whether a management claim about operating-margin direction is supported or contradicted by the displayed revenue and operating-income table.

Companion to *Meridians: Numerical Representations and Structured Answers in Financial Question Answering*, Alina Khay, November 2026. Selected code and recorded-artifact release: 2 November 2026.

## Purpose and scope

This focused synthetic diagnostic supports controlled tests of numerical evidence, claim wording, answer consistency and structured serialization. It is a small template-based evaluation suite, with no real company names or private financial information. It does not represent full financial-QA coverage, real financial advice, an independently human-validated benchmark or 128 independent scenarios.

The final research adapters were trained on selected FinQA/TAT-QA arithmetic questions, **not these synthetic cases**. The earlier synthetic engineering corpus is a different artifact and is not included. FinQA and TAT-QA corpora are not redistributed in this repository.

## Construction and labels

The original synthetic pool was generated with seed 42. The 32 scenarios are selected from its 300 test scenarios by sorting SHA256(`1337:cf:<scenario_id>`). Each selected scenario produces every variant; no model outcome determines inclusion. The generator and 32 numerical seed records are included. Run `python generate.py --check` to verify the published data without network access.

Operating margin equals operating income divided by revenue. A claim of increase is supported if the current margin exceeds the prior margin; a decrease is supported if it is lower. Labels use the exact displayed decimal strings, not latent unrounded values. To reverse the numerical direction, current operating income is set from half the prior margin if the original direction is up, or 1.5 times the prior margin otherwise, rounded to three decimals. Revenue and prior income remain fixed. There are no equal-margin cases.

Each quartet has two supported and two contradicted examples (64 of each overall). Related variants share `scenario_id`. Use this grouping for uncertainty and consistency analyses; do not randomly split variants across training and evaluation or describe the variants as independent observations. No training or validation split is supplied.

## Fields and loading

`id`, `scenario_id`, `variant`, `numeric_flip`, `claim_flip`, `narrative`, `numeric_text`, `question`, `gold_label`, the four exact monetary-value strings, `unit`, `prompt`, and `input_sha256`.

The prompt is the exact pre-chat-template instruction used in the frozen experiment. Input hashes and gold labels match the original evaluation. Model chat-template application is separate. JSONL decimal values are strings to preserve the displayed evidence exactly.

```python
from datasets import load_dataset
ds = load_dataset("alinakhay/Meridians-Counterfactuals", split="test")
```

For the bundled local copy: `load_dataset("json", data_files="data/test.jsonl", split="train")`; the library's default local loader name does not change this artifact's evaluation-only role.

## Evaluation

Report both case accuracy (128 cases) and all-four-correct accuracy (32 scenarios), retaining all cases. The paper's primary strict scorer requires one label in complete answer/scale JSON with an empty scale. The separate, explicitly **post-hoc** scale-agnostic diagnostic accepts a complete two-key JSON object, a single supported/contradicted string or one-element list, and any scale string. It never searches prose or uses gold labels to parse outputs. Invalid responses remain wrong. The diagnostic does not replace the frozen primary scores.

The companion [research explorer](https://huggingface.co/spaces/alinakhay/Meridians-Research-Explorer) displays all ten recorded conditions; it does not perform live model inference. [Text adapter](https://huggingface.co/alinakhay/Meridians-8B-Text-QLoRA) · [Numerical-fusion adapter](https://huggingface.co/alinakhay/Meridians-8B-Numeric-Fusion). The scientific protocol, scorers and all final answer texts are available in the [research-code release](https://github.com/alinakhay/Meridians-research). A real arXiv link will be added only after an identifier exists.

## Limitations and licensing

Templates, one operation family, two periods, a small scenario count and synthetic values limit generalization. No independent human realism review has been completed. Releasing labelled test cases permits training on them later; any such use must be disclosed and cannot support a fresh held-out evaluation on this same suite.

Original synthetic data: CC0-1.0, as declared in the original generation manifest. Original code: MIT (CODE_LICENSE). These terms do not apply to third-party datasets or backbone models.

## Citation

Alina Khay. *Meridians: Numerical Representations and Structured Answers in Financial Question Answering*. Manuscript, November 2026. An arXiv identifier will be added after it is assigned.

[Selected research code and complete final answer record](https://github.com/alinakhay/Meridians-research).
