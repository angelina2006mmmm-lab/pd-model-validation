"""Run and verify a local MLflow experiment over actual PyTorch training artifacts.

This module performs real MLflow logging; use a local file-backed tracking store
(no server, credentials or private document samples).
"""
import argparse,json
from pathlib import Path

def track(metrics_dir="artifacts", store="mlruns", report="reports/mlflow_verified_runs.json"):
    import mlflow
    from mlflow.tracking import MlflowClient
    root=Path(metrics_dir).resolve()
    uri=Path(store).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("docvision-synthetic-cnn-vit")
    summaries=[]
    for architecture in ("cnn","vit"):
        file=root/f"{architecture}_metrics.json"
        if not file.is_file():raise FileNotFoundError(file)
        metrics=json.loads(file.read_text(encoding="utf-8"))
        with mlflow.start_run(run_name=f"{architecture}-documentvision") as run:
            mlflow.set_tags({"project":"document-vision-lab","data_origin":"synthetic","cv_architecture":architecture})
            mlflow.log_params({
                "architecture":architecture,"optimizer":"AdamW","epochs":int(metrics["epochs"]),
                "train_size":int(metrics["train_size"]),
                "validation_size":int(metrics["validation_size"]),"test_size":int(metrics["test_size"]),
                "dataset":"synthetic-only","seed":"42",
            })
            for k in ("test_accuracy","test_macro_f1","best_validation_accuracy",
                      "cpu_latency_ms_p50","cpu_latency_ms_p95","parameters"):
                mlflow.log_metric(k,float(metrics[k]))
            for item in metrics["train_history"]:
                mlflow.log_metric("epoch_train_loss",float(item["train_loss"]),step=int(item["epoch"]))
                mlflow.log_metric("epoch_val_accuracy",float(item["val_accuracy"]),step=int(item["epoch"]))
            mlflow.log_artifact(str(file),artifact_path="metrics")
            summaries.append({"run_id":run.info.run_id,"model":architecture,
                "test_accuracy":float(metrics["test_accuracy"]),
                "test_macro_f1":float(metrics["test_macro_f1"])})
    client=MlflowClient()
    ids={entry["run_id"] for entry in summaries}
    for item in summaries:
        saved=client.get_run(item["run_id"])
        if saved.info.status!="FINISHED":raise AssertionError("MLflow run not finished")
        if saved.data.params["architecture"]!=item["model"]:raise AssertionError("Wrong model")
        if "test_macro_f1" not in saved.data.metrics:raise AssertionError("No logged metric")
        if "epoch_train_loss" not in saved.data.metrics:raise AssertionError("No epoch losses")
    if len(ids)!=2:raise AssertionError("Not two distinct runs")
    result={"tracking_backend":"MLflow local file store","verified_runs":2,
            "runs":summaries,"dataset":"fully synthetic document layouts",
            "verified_by":"MLflowClient.get_run and logged metrics/parameters",
            "notes":"A local experiment store, not a remote managed MLflow service"}
    outfile=Path(report);outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--metrics-dir",default="artifacts")
    parser.add_argument("--store",default="mlruns")
    parser.add_argument("--report",default="reports/mlflow_verified_runs.json")
    a=parser.parse_args()
    track(a.metrics_dir,a.store,a.report)
