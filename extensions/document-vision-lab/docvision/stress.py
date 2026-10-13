"""Image robustness checks on artificial examples, not real bank data."""
from pathlib import Path
import json,numpy as np,torch
from PIL import Image,ImageFilter,ImageEnhance
from .data import CLASSES
from .inference import Predictor
def stress_test(data_root='data',artifact='artifacts/cnn_torchscript.pt',report='reports/stress_test.json'):
 root=Path(data_root);rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='test']
 model=Predictor(artifact).model
 conditions={'original':lambda x:x,'gaussian_blur_2px':lambda x:x.filter(ImageFilter.GaussianBlur(2)),
  'dark_50pct':lambda x:ImageEnhance.Brightness(x).enhance(.5),
  'rotate_12deg':lambda x:x.rotate(12,fillcolor='white')}
 result={}
 for name,transform in conditions.items():
  count=0;errors=[]
  for r in rows:
   im=transform(Image.open(root/r['path']).convert('RGB')).resize((96,64))
   arr=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.
   with torch.inference_mode():i=int(model(torch.from_numpy(arr.copy()).unsqueeze(0)).argmax())
   count+=int(i==CLASSES.index(r['class_name']))
   if i!=CLASSES.index(r['class_name']):errors.append({'case':r['path'],'predicted':CLASSES[i]})
  result[name]={'samples':len(rows),'accuracy':round(count/len(rows),4),'errors':errors[:10]}
 out=Path(report);out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps({'dataset':'synthetic perturbations','results':result,'limitations':'not real documents'},indent=2))
 return result
if __name__=='__main__':print(json.dumps(stress_test(),indent=2))
