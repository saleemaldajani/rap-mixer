import os

import gradio as gr

from rap_mixer.ui.app import CSS, build_app

demo = build_app()

if __name__ == "__main__":
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.queue(default_concurrency_limit=4).launch(
        css=CSS,
        theme=gr.themes.Default(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=server_port,
    )
