class TransformersWhisperTranscriber:
    def __init__(self, model: str = "openai/whisper-small.en"):
        self.model_name = model
        self._pipe = None

    def transcribe_file(self, path: str):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("automatic-speech-recognition", self.model_name)
        return self._pipe(path, return_timestamps="word")["text"]

