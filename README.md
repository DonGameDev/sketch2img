# README.md

This branch contains a revamp prototype for a real-time local sketch-to-image demo.

Key files:
- app.py (fixed original Gradio app + improved helpers)
- frontend/gradio_stream.py (Colab-friendly Gradio demo using diffusers)
- backend/server.py (FastAPI WebSocket scaffold)
- adapters/local_onnx_adapter.py (ONNX adapter scaffold)
- workers/worker.py (cancelable worker scaffold)
- demos/colab_demo.ipynb (Colab notebook to run the Gradio demo on T4)
- docs/hardware_setup_windows.md (Windows DirectML + ONNX quick guide)

See the PR for more details and next steps.
