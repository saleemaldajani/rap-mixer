from rap_mixer.audio.features import AudioFeatureExtractor


class UploadedInstrumentalProvider:
    def analyze(self, audio):
        if audio is None:
            return {"available": False, "warning": "No instrumental supplied"}
        result = AudioFeatureExtractor().analyze_audio(*audio)
        result["available"] = True
        return result

