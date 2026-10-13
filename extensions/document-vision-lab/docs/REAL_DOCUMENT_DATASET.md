# Real document evaluation — provenance, license and methodology

**Source:** [CORD: A Consolidated Receipt Dataset for Post-OCR Parsing](https://github.com/clovaai/cord), public CORD-v2 test images under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The images are **real receipt scans**, not synthetic bank-card, passport, insurance or ID documents.

**Data URL:** [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) — the test Parquet path is pinned in `docvision/real_cord_vlm.py`. Its images and original ground-truth JSON are downloaded by GitHub Actions; **the photos are not committed to this repository**. Public dataset attribution: Park et al., *CORD*, Document Intelligence Workshop, NeurIPS 2019.

**Pretrained model:** [naver-clova-ix/donut-base-finetuned-cord-v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2), pretrained fine-tuned checkpoint released by the model authors (MIT). This project's work is **inference, field parsing and evaluation**; it does not train or fine-tune a Donut/VLM.

**Evaluation protocol:**
- Take the **first 20 test-split receipts** (deterministic order, no selection by model output).
- Extract `gt_parse.total.total_price` from CORD ground truth when present.
- Pass **full scanned image** to pretrained Donut with the `<s_cord-v2>` task prompt, greedy decoding and max 256 tokens; parse structured output through `DonutProcessor.token2json`.
- Run independent Tesseract OCR with PSM 6, then extract a final TOTAL-like line via a transparent baseline regex.
- Normalize expected and predicted monetary strings to digit sequences, compare exact match for annotated receipts only.
- Report eligible count, exact-match ratios, structured JSON parse rate, P50 CPU latency, and per-image successes/errors.
- Keep the actual image bytes out of tracked git and the generated PDF resume.

**Limits:** 20 test receipts is a small exploratory benchmark, especially on the same domain used to tune Donut upstream. Not a confidence-bound generalization claim. No personally identifying document handling, no claims of passport/card recognition, and no throughput or mobile hardware benchmark. If the data/model is unavailable, the workflow fails rather than substituting synthetic outputs.

**Reproduce:** run GitHub Actions workflow `.github/workflows/real-cord-donut-vlm.yml`; it publishes the JSON report as a run artifact. Public model and dataset files must be downloadable.

