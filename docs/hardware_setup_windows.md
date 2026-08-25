# docs/hardware_setup_windows.md

Windows (DirectML + ONNX Runtime) quick setup for AMD GPU acceleration

1. Install Python 3.10+ and create a virtual environment
   python -m venv .venv
   .\.venv\Scripts\activate

2. Install ONNX Runtime with DirectML provider (Windows):
   pip install onnxruntime-directml

3. Install basic Python deps for prototype:
   pip install -r requirements.txt
   pip install diffusers transformers accelerate xformers  # if using diffusers in Colab

4. If you plan to use the NPU (Ryzen NPU), check vendor SDK and ONNX provider from the laptop vendor. If available, install their runtime and register it as an ONNX provider.

5. Run a small ONNX test to confirm GPU provider is available. See the README in the repo for an example test script.
