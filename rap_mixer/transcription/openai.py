class OpenAITranscriber:
    def __init__(self, client):
        self.client = client

    def transcribe_file(self, path: str):
        with open(path, "rb") as audio:
            return self.client.audio.transcriptions.create(model="whisper-1", file=audio).text

