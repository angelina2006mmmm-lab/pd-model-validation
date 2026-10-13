"""Optional MLflow adapter; actual MLflow execution not verified."""
import json
from pathlib import Path
def record_training(metrics_json,tracking_uri='mlruns'):
 import mlflow
 mlflow.set_tracking_uri(tracking_uri)
 mlflow.set_experiment('synthetic-document-type-classification')
 metrics=json.loads(Path(metrics_json).read_text())
 with mlflow.start_run(run_name=metrics['model']):
  mlflow.log_params({'architecture':metrics['model'],'dataset':'synthetic','epochs':metrics['epochs']})
  for key in ('test_accuracy','test_macro_f1','cpu_latency_ms_p50','cpu_latency_ms_p95'):
   mlflow.log_metric(key,metrics[key])
  mlflow.log_artifact(str(metrics_json))
 return True
