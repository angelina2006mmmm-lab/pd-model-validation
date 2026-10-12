# Small transformer baseline — research note

A real PyTorch **TransformerEncoder** is trained from random initialization on six synthetic monthly histories of borrower covariates. Chronological train/validation/OOT split; inputs are generated without looking at the outcome column. This is a proof of training / inference on financial-format sequences, not a pretrained Transformer or LLM.

Validation ROC-AUC: **0.5865**; out-of-time ROC-AUC: **0.5764** on a separate 1,400-row synthetic generator run. It substantially underperforms the stronger structured tabular baseline from another experiment. A sensible next step is richer time series and better feature design, not pretending it is competitive.

Run: `PYTHONPATH=src python -m creditrisk.transformer_demo`.
