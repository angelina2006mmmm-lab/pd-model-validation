"""Versioned PD scoring REST API, synthetic demonstration only."""
import json,logging,time,os
from collections import Counter
from threading import Lock
from uuid import uuid4
from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,ConfigDict
from .model import PDModel

model=PDModel(version=os.getenv('PD_MODEL_VERSION','1.0.0'))
logger=logging.getLogger('risk_model_serving')
logger.setLevel(logging.INFO)
if not logger.handlers:
 h=logging.StreamHandler();h.setFormatter(logging.Formatter('%(message)s'));logger.addHandler(h)
lock=Lock();outcomes=Counter();latencies=[]

class Application(BaseModel):
 model_config=ConfigDict(extra='forbid')
 application_id:str=Field(min_length=2,max_length=40,pattern=r'^[A-Za-z0-9_-]+$')
 debt_to_income:float=Field(ge=0,le=1,allow_inf_nan=False)
 missed_payments_12m:int=Field(ge=0,le=12)
 credit_history_months:int=Field(ge=0,le=600)
 utilization:float=Field(ge=0,le=1,allow_inf_nan=False)

app=FastAPI(title='Credit risk model serving (educational)',version='1.0.0')
@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
 # Omit raw rejected input to prevent NaN serialization and sensitive-data exposure.
 return JSONResponse(status_code=422,content={'detail':[{'loc':[str(x) for x in e.get('loc',[])],'type':e['type'],'msg':e['msg']} for e in exc.errors()]})
@app.middleware('http')
async def request_log(request:Request,call_next):
 start=time.perf_counter();rid=request.headers.get('X-Request-ID','')
 if not(0<len(rid)<=72 and all(c.isalnum() or c in '_-' for c in rid)):rid=str(uuid4())
 status=500
 try:
  response=await call_next(request);status=response.status_code
  response.headers['X-Request-ID']=rid
  return response
 finally:
  ms=round((time.perf_counter()-start)*1000,3)
  with lock:
   outcomes[str(status//100)+'xx']+=1;latencies.append(ms)
  # No application ID, feature values or raw request body logged.
  logger.info(json.dumps({'event':'http','request_id':rid,'route':request.url.path,'status':status,
     'model_version':model.info['version'],'latency_ms':ms}))

@app.get('/health')
def health():return {'status':'ok','model_version':model.info['version']}
@app.get('/v1/model')
def version():return {'model_version':model.info['version'],'features':model.info['features'],
 'training_data':model.info['trained_on'],'test_roc_auc':model.info['test_roc_auc']}
@app.post('/v1/score')
def score(payload:Application):
 p,band=model.predict(payload.model_dump())
 return {'application_id':payload.application_id,'default_probability':round(p,6),
 'risk_band':band,'model_version':model.info['version'],
 'disclaimer':'Synthetic PD estimate; not a bank lending decision'}
@app.get('/metrics')
def metrics():
 with lock:
  return {'requests':len(latencies),'status_groups':dict(outcomes),'mean_latency_ms':round(sum(latencies)/len(latencies),3) if latencies else 0,
   'model_version':model.info['version'],'note':'in-memory demo only; not production telemetry'}
