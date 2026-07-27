class FasterWhisperTranscriber:
    def __init__(self, model: str = "small.en"):
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device="auto", compute_type="int8")

    def transcribe_file(self, path: str):
        self._load()
        segments, _ = self._model.transcribe(path, word_timestamps=True)
        return " ".join(x.text.strip() for x in segments)

