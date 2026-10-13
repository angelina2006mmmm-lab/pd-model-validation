"""Preview a tiny sample of licensed real CORD-v2 receipts via HF dataset viewer API."""
import json,requests
API="https://datasets-server.huggingface.co"
NAME="naver-clova-ix/cord-v2"
def get(path,params):
 r=requests.get(API+path,params=params,timeout=100)
 print("request:",r.url,"status:",r.status_code)
 r.raise_for_status()
 return r.json()
def main():
 s=get("/splits",{"dataset":NAME})
 splits=[x for x in s.get("splits",[]) if x["split"]=="test"]
 print("splits",[(x.get("config"),x.get("split")) for x in splits])
 if not splits:raise ValueError("No public CORD test split")
 config=splits[0]["config"]
 data=get("/rows",{"dataset":NAME,"config":config,"split":"test","offset":0,"length":2})
 print("root keys:",list(data.keys()))
 for elem in data.get("rows",[]):
  row=elem["row"]
  print("row_idx",elem["row_idx"],"keys",list(row))
  print("image type",type(row.get("image")).__name__,"image preview",str(row.get("image"))[:350])
  g=row.get("ground_truth")
  print("gt type",type(g).__name__,"gt prefix",str(g)[:1100])
 if not data.get("rows"):raise ValueError("No rows")
if __name__=="__main__":main()
