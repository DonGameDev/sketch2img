"""
workers/worker.py
A very small cancelable worker loop used by backend to run generation tasks.
"""
import threading
import time

class GenerationWorker:
    def __init__(self, adapter):
        self.adapter = adapter
        self.current_token = None
        self.lock = threading.Lock()

    def submit_preview(self, conditioning_image, prompt, callback):
        with self.lock:
            if self.current_token:
                self.current_token.cancel()
            token = CancelToken()
            self.current_token = token
        def _run():
            result = self.adapter.generate_preview(conditioning_image, prompt, token=token)
            callback(result)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

# Note: CancelToken is defined in adapters/local_onnx_adapter.py; import there when wiring up.
