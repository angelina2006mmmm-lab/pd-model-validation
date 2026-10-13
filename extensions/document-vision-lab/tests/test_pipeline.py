import io
import pytest,torch
from PIL import Image
from fastapi.testclient import TestClient
from docvision.data import CLASSES,draw_document,generate,validate_dataset,DocumentDataset
from docvision.models import get_model
from docvision.train import train_and_evaluate
from docvision.inference import preprocess_image,Predictor,create_app
def png(im):
 f=io.BytesIO();im.save(f,format='PNG');return f.getvalue()
def test_dimensions():
 for i in range(3):assert draw_document(i,2026).size==(96,64)
def test_balanced_splits(tmp_path):
 generate(tmp_path,(3,2,2))
 assert validate_dataset(tmp_path)['files']==21
 assert len(DocumentDataset(tmp_path,'train'))==9
def test_corrupt_detect(tmp_path):
 m=generate(tmp_path,(1,1,1))
 (tmp_path/m[0]['path']).write_bytes(b'bad')
 with pytest.raises(ValueError):validate_dataset(tmp_path)
def test_duplicate_detect(tmp_path):
 m=generate(tmp_path,(2,1,1))
 (tmp_path/m[1]['path']).write_bytes((tmp_path/m[0]['path']).read_bytes())
 with pytest.raises(ValueError):validate_dataset(tmp_path)
@pytest.mark.parametrize('name',['cnn','vit'])
def test_model_shape(name):assert get_model(name)(torch.rand(2,3,64,96)).shape==(2,3)
def test_bad_model():
 with pytest.raises(ValueError):get_model('broken')
def test_image_preprocessing():assert preprocess_image(png(draw_document(0,7))).shape==(1,3,64,96)
@pytest.mark.parametrize('payload',[b'',b'broken',b'x'*(2*1024*1024+1)])
def test_image_negative(payload):
 with pytest.raises(ValueError):preprocess_image(payload)
def test_model_export_api(tmp_path):
 generate(tmp_path/'data',(7,3,3),seed=3)
 out=train_and_evaluate(tmp_path/'data',tmp_path/'out','cnn',epochs=1,batch_size=8,seed=3)
 assert out['test_size']==9
 model=tmp_path/'out/cnn_torchscript.pt'
 assert Predictor(model).predict(png(draw_document(1,55)))['document_type'] in CLASSES
 client=TestClient(create_app(model))
 assert client.get('/health').status_code==200
 assert client.post('/predict',content=png(draw_document(1,55)),headers={'Content-Type':'image/png'}).status_code==200
 assert client.post('/predict',content=b'bad',headers={'Content-Type':'image/png'}).status_code==422
 assert client.post('/predict',content=b'bad',headers={'Content-Type':'text/plain'}).status_code==415
def test_transformer_export():
 m=get_model('vit').eval();x=torch.rand(1,3,64,96)
 with torch.inference_mode():assert torch.allclose(m(x),torch.jit.trace(m,x,check_trace=False)(x),atol=1e-4)
def test_stress(tmp_path):
 from docvision.stress import stress_test
 generate(tmp_path/'data',(2,2,2))
 train_and_evaluate(tmp_path/'data',tmp_path/'out','cnn',epochs=1,batch_size=4)
 r=stress_test(tmp_path/'data',tmp_path/'out/cnn_torchscript.pt',tmp_path/'stress.json')
 assert r['original']['samples']==6 and 'dark_50pct' in r
