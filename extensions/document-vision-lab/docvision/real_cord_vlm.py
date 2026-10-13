"""Real CORD-v2 receipt images + pretrained Donut vision-language parsing.

Licensed dataset CC-BY-4.0; only metrics (not receipt images) are published.
This is fine-tuned pretrained VLM *inference*, not model fine-tuning done here.
"""
import io,json,re,time,statistics,argparse
from pathlib import Path
from PIL import Image
DATASET="naver-clova-ix/cord-v2"
PARQUET_FILE="data/test-00000-of-00001-9c204eb3f4e11791.parquet"
VLM="naver-clova-ix/donut-base-finetuned-cord-v2"
def rows(limit=3):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    if not 1<=limit<=25:raise ValueError("limit must be 1..25")
    f=hf_hub_download(repo_id=DATASET,repo_type="dataset",filename=PARQUET_FILE)
    reader=pq.ParquetFile(f)
    count=0
    for batch in reader.iter_batches(batch_size=1):
        item=batch.to_pylist()[0]
        gt=json.loads(item["ground_truth"])
        image=Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")
        yield image,gt
        count+=1
        if count>=limit:break
def scalar(value):
    if isinstance(value,list):
        return scalar(value[0]) if value else ""
    if value is None:return ""
    return str(value).strip()
def norm_price(value):
    """Canonical digits for form-field strict equality; no financial inference."""
    return re.sub(r"[^0-9]","",scalar(value))
def gt_total(ground_truth):
    parse=ground_truth["gt_parse"]
    total=parse.get("total",{})
    if isinstance(total,list):total=total[0] if total else {}
    return norm_price(total.get("total_price",""))
def parse_total(structured):
    if not isinstance(structured,dict):return ""
    total=structured.get("total",{})
    if isinstance(total,list):total=total[0] if total else {}
    return norm_price(total.get("total_price","")) if isinstance(total,dict) else ""
def init_donut():
    import torch
    from transformers import DonutProcessor,VisionEncoderDecoderModel
    torch.set_num_threads(2);torch.manual_seed(42)
    proc=DonutProcessor.from_pretrained(VLM)
    model=VisionEncoderDecoderModel.from_pretrained(VLM,low_cpu_mem_usage=False).eval()
    return proc,model
def infer_donut(image,proc,model):
    import torch
    pixels=proc(image,return_tensors="pt").pixel_values
    prefix=proc.tokenizer("<s_cord-v2>",add_special_tokens=False,return_tensors="pt").input_ids
    with torch.inference_mode():
        out=model.generate(pixels,decoder_input_ids=prefix,max_length=256,
          pad_token_id=proc.tokenizer.pad_token_id,
          eos_token_id=proc.tokenizer.eos_token_id,
          num_beams=1,use_cache=True,
          bad_words_ids=[[proc.tokenizer.unk_token_id]])
    generated=proc.batch_decode(out,skip_special_tokens=False)[0]
    generated=generated.replace(proc.tokenizer.eos_token,"").replace(proc.tokenizer.pad_token,"")
    generated=re.sub(r"<.*?>","",generated,count=1).strip()
    try:
        prediction=proc.token2json(generated)
    except Exception:
        prediction={}
    return prediction,generated[:450]
def benchmark(n=3,output="reports/cord_real_receipt_donut.json"):
    pairs=list(rows(n))
    proc,model=init_donut()
    result=[]
    for idx,(image,gt) in enumerate(pairs):
        start=time.perf_counter()
        parsed,raw=infer_donut(image,proc,model)
        elapsed=(time.perf_counter()-start)*1000
        expected=gt_total(gt);predicted=parse_total(parsed)
        result.append({"image_index":idx,"expected_total_digits":expected,
           "predicted_total_digits":predicted,
           "total_exact_match":bool(expected) and expected==predicted,
           "has_structured_output":bool(parsed),
           "latency_ms":round(elapsed,2),"prediction_example":raw[:120]})
    total=len(result);latencies=sorted(x["latency_ms"] for x in result)
    summary={
     "dataset":"CORD-v2 test split, genuine publicly released receipt images",
     "dataset_license":"CC-BY-4.0","model":VLM,"model_task":"pretrained Donut image-to-structured-text VLM inference",
     "method":"offline evaluation, deterministic prefix decoding; no fine-tuning in project",
     "receipts_evaluated":total,
     "receipts_with_groundtruth_total":sum(bool(x["expected_total_digits"]) for x in result),
     "total_price_exact_match_rate":round(sum(x["total_exact_match"] for x in result)/total,4),
     "structured_output_rate":round(sum(x["has_structured_output"] for x in result)/total,4),
     "p50_latency_ms":round(statistics.median(latencies),2),
     "individual_cases":result,
     "limitations":"Tiny sample; receipt domain only, no bank card/passport OCR; CPU benchmark not production SLA. Pretrained weights, not trained in project."
    }
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--n",type=int,default=3)
    a.add_argument("--output",default="reports/cord_real_receipt_donut.json")
    x=a.parse_args();benchmark(x.n,x.output)
