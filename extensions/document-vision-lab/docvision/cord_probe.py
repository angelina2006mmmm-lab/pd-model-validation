"""Inspect a licensed CORD-v2 test Parquet directly if the viewer is unavailable.

Actual dataset bytes are downloaded in Actions and never committed to git.
"""
import json,os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
DATASET="naver-clova-ix/cord-v2"
PATH="data/test-00000-of-00001-9c204eb3f4e11791.parquet"
def main():
    file=hf_hub_download(repo_id=DATASET,repo_type="dataset",filename=PATH)
    print("downloaded_size_MB",round(os.stat(file).st_size/1e6,2))
    p=pq.ParquetFile(file)
    print("parquet schema:",p.schema)
    for batch in p.iter_batches(batch_size=1):
        row=batch.to_pylist()[0]
        print("keys",list(row))
        for k,v in row.items():
            if isinstance(v,dict):
                print(k,"dict keys",list(v),"brief",str({a:len(b) if isinstance(b,bytes) else str(b)[:170] for a,b in v.items()})[:700])
            elif isinstance(v,bytes):
                print(k,"bytes",len(v))
            else:
                print(k,"type",type(v).__name__,"prefix",str(v)[:1800])
        break
if __name__=="__main__":main()
