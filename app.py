import os

try:
    # Present only on Hugging Face ZeroGPU Spaces (installed by the platform).
    import spaces
except ImportError:
    spaces = None

import gradio as gr

from rap_mixer.ui.app import CSS, build_app

if spaces is not None:

    @spaces.GPU(duration=10)
    def zerogpu_startup_probe() -> bool:
        """ZeroGPU hosting requires one @spaces.GPU function at startup.

        The Rap Mixer's deterministic analysis path is CPU-only, so this
        probe is never called by the UI; the decorator is effect-free
        outside ZeroGPU environments.
        """
        return True


demo = build_app()

if __name__ == "__main__":
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    # Hugging Face Spaces and other containers need a public bind; local runs stay loopback.
    default_host = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    demo.queue(default_concurrency_limit=4).launch(
        css=CSS,
        theme=gr.themes.Default(),
        server_name=os.getenv("GRADIO_SERVER_NAME", default_host),
        server_port=server_port,
    )
