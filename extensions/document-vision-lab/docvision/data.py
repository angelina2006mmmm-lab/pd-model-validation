"""Synthetic document layouts only. No bank documents or personal data."""
import hashlib,json,random
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageEnhance,ImageFont
CLASSES=('bank_card','passport_page','driver_license')
WIDTH,HEIGHT=96,64
def _font(size):
 p='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 return ImageFont.truetype(p,size) if Path(p).exists() else ImageFont.load_default()
def draw_document(label,seed):
 rng=random.Random(seed);w,h=WIDTH,HEIGHT
 if label==0:
  im=Image.new('RGB',(w,h),(rng.randrange(30,65),rng.randrange(75,145),rng.randrange(130,195)));d=ImageDraw.Draw(im)
  d.rounded_rectangle((3,4,w-4,h-5),radius=9,outline=(225,235,245),width=2)
  d.rounded_rectangle((13,17,31,31),radius=3,fill=(218,193,110),outline=(91,71,40))
  d.line((13,24,31,24),fill=(91,71,40))
  d.text((14,36),f'{rng.randrange(1000,9999)}  {rng.randrange(1000,9999)}',font=_font(9),fill='white')
  d.rectangle((68,10,82,18),fill=(245,245,235))
 elif label==1:
  im=Image.new('RGB',(w,h),(rng.randrange(207,234),rng.randrange(190,219),rng.randrange(162,196)));d=ImageDraw.Draw(im)
  d.rectangle((3,4,w-4,h-5),outline=(105,87,67),width=2)
  d.rectangle((9,14,35,47),fill=(rng.randrange(65,140),120,110),outline=(44,45,42))
  d.ellipse((15,18,29,31),fill=(221,182,146));d.rectangle((15,31,29,42),fill=(83,101,114))
  for y in (16,24,32,40):d.line((42,y,86-rng.randrange(0,14),y),fill=(91,79,69),width=2)
  d.text((20,50),'ID',fill=(78,65,56),font=_font(8))
 elif label==2:
  im=Image.new('RGB',(w,h),(rng.randrange(190,215),rng.randrange(212,235),rng.randrange(211,236)));d=ImageDraw.Draw(im)
  d.rounded_rectangle((3,5,w-4,h-5),radius=3,outline=(55,90,95),width=2)
  d.rectangle((5,7,91,17),fill=(rng.randrange(65,130),90,115))
  d.rectangle((67,22,86,50),fill=(112,124,134))
  d.ellipse((71,24,82,36),fill=(216,184,153));d.rectangle((70,37,84,47),fill=(35,49,69))
  for y in (24,31,38,45):d.line((12,y,53-rng.randrange(0,12),y),fill=(68,92,96),width=2)
  d.text((13,52),'DL',fill=(72,91,98),font=_font(8))
 else:raise ValueError('Unknown class')
 return ImageEnhance.Brightness(im).enhance(rng.uniform(.85,1.15))
def generate(root,per_class=(150,35,35),seed=42):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for split,n in zip(('train','val','test'),per_class):
  for cls,name in enumerate(CLASSES):
   (root/split/name).mkdir(parents=True,exist_ok=True)
   for i in range(n):
    s=seed+cls*100000+{'train':0,'val':20000,'test':40000}[split]+i
    path=root/split/name/f'{name}_{s}.png';draw_document(cls,s).save(path)
    manifest.append({'path':str(path.relative_to(root)),'split':split,'class_name':name,'source':'synthetic','seed':s})
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
 return manifest
def validate_dataset(root):
 root=Path(root);records=json.loads((root/'manifest.json').read_text());counts=Counter();hashes=set();bad=[]
 for row in records:
  p=root/row['path']
  try:
   with Image.open(p) as im:
    im.load()
    if im.size!=(WIDTH,HEIGHT) or im.mode!='RGB':bad.append(row['path'])
   dg=hashlib.sha256(p.read_bytes()).hexdigest()
   if dg in hashes:bad.append(row['path']+' duplicate')
   hashes.add(dg);counts[(row['split'],row['class_name'])]+=1
  except Exception:bad.append(row['path']+' invalid')
 if bad:raise ValueError('Invalid images: '+str(bad[:10]))
 if any(counts.get((s,c),0)==0 for s in ('train','val','test') for c in CLASSES):raise ValueError('Missing class')
 return {'files':len(records),'counts':{f'{s}/{c}':n for (s,c),n in counts.items()},'invalid':0,'duplicates':0,'privacy':'synthetic'}
class DocumentDataset:
 def __init__(self,root,split):
  import numpy as np,torch
  self.root=Path(root);self.np=np;self.torch=torch
  self.records=[r for r in json.loads((self.root/'manifest.json').read_text()) if r['split']==split]
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  r=self.records[i]
  with Image.open(self.root/r['path']) as im:a=self.np.asarray(im.convert('RGB'),dtype=self.np.float32).transpose(2,0,1)/255.
  return self.torch.from_numpy(a.copy()),CLASSES.index(r['class_name'])
