"""TorchScript inference with image checks and FastAPI endpoints."""
import io
import numpy as np,torch
from PIL import Image,ImageOps,UnidentifiedImageError
from .data import CLASSES,WIDTH,HEIGHT
MAX_BYTES=2*1024*1024
def preprocess_image(data):
 if not data or len(data)>MAX_BYTES:raise ValueError("Empty or oversized image")
 try:
  with Image.open(io.BytesIO(data)) as im:
   im=ImageOps.exif_transpose(im).convert('RGB').resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR)
   arr=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.
 except (UnidentifiedImageError,OSError,ValueError) as e:raise ValueError("Invalid image") from e
 return torch.from_numpy(arr.copy()).unsqueeze(0)
class Predictor:
 def __init__(self,artifact):self.model=torch.jit.load(str(artifact),map_location='cpu').eval()
 @torch.inference_mode()
 def predict(self,data):
  p=torch.softmax(self.model(preprocess_image(data)),dim=-1)[0]
  i=int(p.argmax());return {'document_type':CLASSES[i],'confidence':round(float(p[i]),5),
                            'class_probabilities':{k:round(float(v),5) for k,v in zip(CLASSES,p)}}
def create_app(artifact):
 from fastapi import FastAPI,HTTPException,Request
 app=FastAPI(title="Educational document classifier")
 pred=Predictor(artifact)
 @app.get('/health')
 def health():return {'status':'ok','model':'synthetic_document_type_classifier'}
 @app.post('/predict')
 async def predict(req:Request):
  if req.headers.get('content-type','').split(';')[0] not in ('image/png','image/jpeg','image/webp'):
   raise HTTPException(415,"Expected image")
  try:return pred.predict(await req.body())
  except ValueError as e:raise HTTPException(422,str(e))
 return app
