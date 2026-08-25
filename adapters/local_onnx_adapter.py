"""
adapters/local_onnx_adapter.py
A scaffold for an ONNX-based adapter. This file will load an ONNX model and
expose a simple generate_preview / generate_refine API with cancellation support.
Note: actual implementation depends on which ONNX model you convert for Colab/Windows.
"""

import threading
import time

class CancelToken:
    def __init__(self):
        self._cancelled = False
    def cancel(self):
        self._cancelled = True
    def is_cancelled(self):
        return self._cancelled

class LocalONNXAdapter:
    def __init__(self, model_path=None, provider=None):
        self.model_path = model_path
        self.provider = provider
        # TODO: load the ONNX model using onnxruntime.InferenceSession
        self.session = None

    def load(self):
        # placeholder: load session
        return True

    def generate_preview(self, conditioning_image, prompt, steps=10, token: CancelToken = None):
        # placeholder: emulate work and check cancellation
        for i in range(steps):
            if token and token.is_cancelled():
                return None
            time.sleep(0.05)
        # return a dummy result (in practice, return PIL.Image)
        return None

    def generate_refine(self, conditioning_image, prompt, steps=30, token: CancelToken = None):
        for i in range(steps):
            if token and token.is_cancelled():
                return None
            time.sleep(0.05)
        return None
