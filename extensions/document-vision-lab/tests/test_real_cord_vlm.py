from docvision.real_cord_vlm import norm_price,gt_total,parse_total,scalar
def test_price_normalization():
 assert norm_price("60.000")=="60000"
 assert norm_price("Rp 1,200.00")=="120000"
def test_ground_truth_list():
 assert gt_total({"gt_parse":{"total":{"total_price":"60.000"}}})=="60000"
 assert parse_total({"total":[{"total_price":"60.000"}]})=="60000"
def test_missing_is_empty():
 assert parse_total({})=="" and scalar(None)==""
