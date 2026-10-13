"""Pretrained Tesseract OCR on fully synthetic forms. Not a VLM or bank OCR."""
import json,random,re,statistics,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def make_form(seed):
 r=random.Random(seed);truth={'doc_id':f'{r.randrange(100000,999999)}','total':str(r.randrange(1000,9999))}
 im=Image.new('RGB',(720,300),'white');d=ImageDraw.Draw(im)
 title=ImageFont.truetype(FONT,24);body=ImageFont.truetype(FONT,38)
 d.rectangle((20,20,700,280),outline=(40,60,80),width=3)
 d.text((40,30),'SYNTHETIC TEST FORM',fill=(45,65,75),font=title)
 d.text((45,100),'DOC ID: '+truth['doc_id'],fill=(15,15,15),font=body)
 d.text((45,180),'TOTAL: '+truth['total'],fill=(15,15,15),font=body)
 return im,truth
def extract_fields(img):
 import pytesseract
 raw=pytesseract.image_to_string(img,config='--psm 6')
 a=re.search(r'DOC\s*ID\s*:\s*(\d{6})',raw,re.I)
 b=re.search(r'TOTAL\s*:\s*(\d{4})',raw,re.I)
 return {'doc_id':a.group(1) if a else None,'total':b.group(1) if b else None},raw
def benchmark(n=20,report='reports/ocr_synthetic_benchmark.json'):
 if n<=0:raise ValueError('Positive n only')
 right=0;total=0;dur=[];errors=[]
 for i in range(n):
  image,truth=make_form(500+i)
  start=time.perf_counter();got,text=extract_fields(image);dur.append((time.perf_counter()-start)*1000)
  for key in ('doc_id','total'):total+=1;right+=int(got[key]==truth[key])
  if got!=truth:errors.append({'seed':500+i,'expected':truth,'predicted':got,'ocr':text[:180]})
 result={'source':'clean synthetic printed forms','engine':'pretrained Tesseract --psm 6',
  'forms':n,'fields':total,'field_exact_match':round(right/total,4),
  'p50_latency_ms':round(statistics.median(dur),2),'failures':errors,
  'limitations':'not real ID documents, credit cards, OCR models training, or VLM'}
 dest=Path(report);dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(benchmark(),indent=2))
