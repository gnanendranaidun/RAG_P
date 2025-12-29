
import whisper
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class ASRExtractor:
    def __init__(self, model_name="base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)

    def extract(self, video_path):
        """
        Extracts speech from video and returns a list of segments with timestamps.
        """
        print(f"Transcribing audio from {video_path}...")
        try:
            result = self.model.transcribe(video_path)
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            return segments
        except Exception as e:
            print(f"ASR Warning: Could not transcribe audio (might be silent or missing track). Error: {e}")
            return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python asr.py <video_path>")
    else:
        extractor = ASRExtractor()
        print(extractor.extract(sys.argv[1]))
