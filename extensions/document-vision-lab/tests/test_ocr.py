from docvision.ocr import make_form,extract_fields,benchmark
def test_synthetic_form():
 image,truth=make_form(33)
 assert image.size==(720,300)
 assert len(truth['doc_id'])==6
def test_tesseract_extracts_fields():
 img,truth=make_form(100)
 got,_=extract_fields(img)
 assert got==truth
def test_benchmark(tmp_path):
 result=benchmark(n=3,report=tmp_path/'ocr.json')
 assert result['field_exact_match']>=.8
 assert (tmp_path/'ocr.json').exists()
