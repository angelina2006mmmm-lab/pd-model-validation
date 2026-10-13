"""Train on synthetic images, select by validation, test only after selection."""
import argparse,json,random,statistics,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from .data import CLASSES,DocumentDataset,generate,validate_dataset
from .models import get_model
def train_and_evaluate(data_dir,output_dir,model_name='cnn',epochs=5,batch_size=48,seed=42):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
 data_dir=Path(data_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 if not (data_dir/'manifest.json').exists():generate(data_dir,seed=seed)
 qa=validate_dataset(data_dir)
 train=DataLoader(DocumentDataset(data_dir,'train'),batch_size=batch_size,shuffle=True)
 val=DataLoader(DocumentDataset(data_dir,'val'),batch_size=batch_size)
 test=DataLoader(DocumentDataset(data_dir,'test'),batch_size=batch_size)
 model=get_model(model_name);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
 lossfn=nn.CrossEntropyLoss();history=[];best=-1.;state=None
 for ep in range(1,epochs+1):
  model.train();losses=[]
  for x,y in train:
   opt.zero_grad();loss=lossfn(model(x),y);loss.backward();opt.step();losses.append(float(loss.detach()))
  model.eval();correct=total=0
  with torch.inference_mode():
   for x,y in val:
    p=model(x).argmax(1);correct+=int((p==y).sum());total+=len(y)
  acc=correct/total;history.append({'epoch':ep,'train_loss':round(statistics.mean(losses),5),'val_accuracy':round(acc,5)})
  if acc>best:
   best=acc;state={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state);model.eval();actual=[];pred=[]
 with torch.inference_mode():
  for x,y in test:pred.extend(model(x).argmax(1).tolist());actual.extend(y.tolist())
 classes={}
 for idx,name in enumerate(CLASSES):
  tp=sum(a==idx and b==idx for a,b in zip(actual,pred))
  fp=sum(a!=idx and b==idx for a,b in zip(actual,pred))
  fn=sum(a==idx and b!=idx for a,b in zip(actual,pred))
  classes[name]={'precision':round(tp/(tp+fp) if tp+fp else 0,4),
     'recall':round(tp/(tp+fn) if tp+fn else 0,4),
     'f1':round(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,4),
     'support':sum(a==idx for a in actual)}
 matrix=[[sum(a==i and b==j for a,b in zip(actual,pred)) for j in range(3)] for i in range(3)]
 torch.save(state,output_dir/f'{model_name}_weights.pt')
 traced=torch.jit.trace(model,torch.rand(1,3,64,96),check_trace=False)
 traced.save(str(output_dir/f'{model_name}_torchscript.pt'))
 sample=torch.rand(1,3,64,96);dur=[]
 with torch.inference_mode():
  for _ in range(8):traced(sample)
  for _ in range(40):
   t=time.perf_counter();traced(sample);dur.append((time.perf_counter()-t)*1000)
 dur.sort()
 result={'model':model_name,'dataset':'synthetic_only','dataset_quality':qa,
  'train_size':len(train.dataset),'validation_size':len(val.dataset),'test_size':len(test.dataset),
  'best_validation_accuracy':round(best,4),
  'test_accuracy':round(sum(a==b for a,b in zip(actual,pred))/len(actual),4),
  'test_macro_f1':round(sum(v['f1'] for v in classes.values())/3,4),
  'per_class':classes,'confusion_matrix':matrix,
  'parameters':sum(p.numel() for p in model.parameters()),'epochs':epochs,
  'train_history':history,'cpu_latency_ms_p50':round(statistics.median(dur),3),
  'cpu_latency_ms_p95':round(dur[int(.95*(len(dur)-1))],3),
  'torchscript_model':f'{model_name}_torchscript.pt',
  'notes':'Synthetic layouts only, no actual OCR or document VLM. CPU-specific timings.'}
 (output_dir/f'{model_name}_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--out',default='artifacts')
 p.add_argument('--model',choices=['cnn','vit'],default='cnn');p.add_argument('--epochs',type=int,default=5)
 a=p.parse_args();print(json.dumps(train_and_evaluate(a.data,a.out,a.model,epochs=a.epochs),indent=2))
