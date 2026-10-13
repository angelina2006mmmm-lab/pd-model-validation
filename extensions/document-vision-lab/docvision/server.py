"""Run with uvicorn docvision.server:app after CNN training."""
from pathlib import Path
from .inference import create_app
app=create_app(Path(__file__).resolve().parents[1]/'artifacts/cnn_torchscript.pt')
