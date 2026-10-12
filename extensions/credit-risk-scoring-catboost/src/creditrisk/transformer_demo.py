"""Small PyTorch TransformerEncoder trained on six synthetic financial-history observations."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from .data import create_applications,split_time

def sequence_features(frame,seed=31):
 ids=frame["application_id"].str.replace("SYN-","",regex=False).astype(int).to_numpy()
 samples=[]
 for idx,row in enumerate(frame.itertuples(index=False)):
  rng=np.random.default_rng(seed+int(ids[idx]))
  seq=np.column_stack([
    np.clip(row.utilization+rng.normal(0,.09,6),0,1),
    np.clip(row.dti+rng.normal(0,.06,6),0,1),
    np.clip(row.late_payments_12m/6+rng.normal(0,.07,6),0,1)]).astype("float32")
  samples.append(seq)
 return torch.tensor(np.stack(samples))

class FinancialHistoryTransformer(nn.Module):
 def __init__(self):
  super().__init__()
  self.projection=nn.Linear(3,16)
  self.positional=nn.Parameter(torch.zeros(1,6,16))
  layer=nn.TransformerEncoderLayer(d_model=16,nhead=2,dim_feedforward=32,dropout=0,batch_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
  self.head=nn.Linear(16,1)
 def forward(self,x):
  return self.head(self.encoder(self.projection(x)+self.positional).mean(dim=1)).squeeze(-1)

def train_demo(n=1400,seed=42,epochs=7,output_dir=None):
 torch.set_num_threads(2);torch.manual_seed(seed)
 train,val,oot=split_time(create_applications(n=n,seed=seed))
 model=FinancialHistoryTransformer()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
 X=sequence_features(train)
 y=torch.tensor(train.default_12m.to_numpy(),dtype=torch.float32)
 for _ in range(epochs):
  model.train()
  for ids in torch.randperm(len(X)).split(128):
   logits=model(X[ids])
   loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
   optimizer.zero_grad();loss.backward();optimizer.step()
 model.eval()
 with torch.no_grad():
  pv=model(sequence_features(val)).sigmoid().numpy()
  po=model(sequence_features(oot)).sigmoid().numpy()
 result={"experiment":"PyTorch TransformerEncoder, 6 synthetic monthly financial-history steps",
         "data_note":"Inputs built from application-time covariates only, no target leakage",
         "train_rows":len(train),"validation_rows":len(val),"oot_rows":len(oot),
         "epochs":epochs,"seed":seed,
         "validation_roc_auc":round(float(roc_auc_score(val.default_12m,pv)),4),
         "oot_roc_auc":round(float(roc_auc_score(oot.default_12m,po)),4),
         "limitations":"Tiny random-initialized network; no pretrained model and no real financial outcomes"}
 if output_dir:
  p=Path(output_dir);p.mkdir(parents=True,exist_ok=True)
  (p/"transformer_demo.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 return result
if __name__=="__main__":
 print(json.dumps(train_demo(output_dir=Path(__file__).resolve().parents[2]/"reports"),indent=2))
