from typing import Protocol


class Transcriber(Protocol):
    def transcribe_file(self, path: str): ...
    def transcribe_chunk(self, sample_rate: int, audio): ...

