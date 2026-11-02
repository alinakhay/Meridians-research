---
title: Meridians Research Explorer
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
license: mit
short_description: Recorded financial QA and counterfactual results
models:
- alinakhay/Meridians-8B-Text-QLoRA
- alinakhay/Meridians-8B-Numeric-Fusion
datasets:
- alinakhay/Meridians-Counterfactuals
tags:
- finance
- research
- evaluation
- counterfactual
---
# Meridians Research Explorer

Interactive companion to *Meridians: Numerical Representations and Structured Answers in Financial Question Answering*, Alina Khay, November 2026. The selected code and recorded-artifact release is dated 2 November 2026.

This is a **recorded-results explorer**. It displays the 128 synthetic counterfactual cases, all 1,280 corresponding original answers across ten conditions, and financial-QA aggregate results for every condition. It does not run live model inference or present stored predictions as new answers. The study contains 2,560 final predictions overall; the curated code release retains every answer text. Original FinQA/TAT-QA question corpora are not bundled here.

The explorer preserves the frozen strict scores and labels the separate scale-agnostic diagnostic as **post-hoc**. Synthetic quartet results describe 32 related scenarios, not 128 independent scenarios. All three predetermined seeds are available; the base model was evaluated once. Seed 42 model releases were predetermined, not selected for test performance.

Open `index.html` locally to inspect the app; all assets and recorded predictions are bundled, with no external scripts, model APIs or credentials. The interface offers a scenario selector, independent numeric/claim flips, seed selection, all recorded answers, a full condition table, figures and the accompanying paper. The `dataset/` sibling in the publication bundle contains the exact-data generator and provenance.

Original presentation code is MIT. The included original synthetic cases retain CC0-1.0 terms (DATA_LICENSE). The scientific manuscript and its original figures are CC BY 4.0, separate from the MIT presentation-code license; see the research repository's `paper/RIGHTS.md`. Existing model and upstream dataset terms remain separate. An arXiv identifier will be added after it is assigned.

## Research interface

The interface uses a persistent study navigation, an overview of the controlled comparison, a four-variant counterfactual workspace, and a dedicated paper/figure browser. Model cards show the parsed diagnostic label, the actual scale field, frozen strict and post-hoc correctness, and the unchanged original response on expansion. Changed claims and numerical cells are highlighted. Scenario, seed and flip settings can be restored from a copied case link.

The results chart supports five recorded measures and either all-seed means or each predetermined seed. The full ten-condition table always remains visible below it; its CSV export retains all five scores at the recorded precision. Six figure views are available in the methods gallery, with paired uncertainty in the results view. All content uses the existing frozen records; no new model answers are generated.

[Selected research code and complete final answer record](https://github.com/alinakhay/Meridians-research).
