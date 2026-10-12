import torch
from creditrisk.data import create_applications
from creditrisk.transformer_demo import sequence_features,FinancialHistoryTransformer,train_demo
def test_sequence_without_target_leakage():
 df=create_applications(45)
 before=sequence_features(df)
 df["default_12m"]=1-df["default_12m"]
 after=sequence_features(df)
 assert torch.equal(before,after) and tuple(before.shape)==(45,6,3)
def test_transformer_forward():
 assert FinancialHistoryTransformer()(torch.zeros(3,6,3)).shape==(3,)
def test_training_produces_metrics(tmp_path):
 out=train_demo(n=900,epochs=3,output_dir=tmp_path)
 assert .4<=out["oot_roc_auc"]<=1
 assert (tmp_path/"transformer_demo.json").exists()
