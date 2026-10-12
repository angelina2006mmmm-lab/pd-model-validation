from creditrisk.boosting_challengers import build_models,run_benchmark
from creditrisk.data import FEATURES
def test_models_and_features():
 assert set(build_models(rounds=5))=={"LogisticRegression","XGBoost","LightGBM"}
 assert "default_12m" not in FEATURES and "application_id" not in FEATURES and "application_date" not in FEATURES
def test_reproducible_challenger_benchmark(tmp_path):
 a=run_benchmark(n=1200,rounds=18,output_dir=tmp_path)
 b=run_benchmark(n=1200,rounds=18)
 assert a["selected_model"]==b["selected_model"]
 assert (tmp_path/"boosting_comparison.md").exists()
 assert (tmp_path/"boosting_comparison.json").exists()
 for name in ("XGBoost","LightGBM"):
  assert .5<a["oot"][name]["roc_auc"]<1
  assert a["validation"][name]["roc_auc"]==b["validation"][name]["roc_auc"]
