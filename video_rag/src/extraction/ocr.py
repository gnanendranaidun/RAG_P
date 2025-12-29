
import easyocr
import cv2
import numpy as np

class OCRExtractor:
    def __init__(self, languages=['en'], gpu=True):
        print(f"Loading EasyOCR reader for {languages} (GPU={gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract(self, video_path, interval_seconds=1):
        """
        Extracts text from video frames sampled at interval_seconds.
        """
        print(f"Extracting OCR from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        results = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                # EasyOCR handles numpy arrays (OpenCV format) directly
                detections = self.reader.readtext(frame)
                
                texts = []
                for bbox, text, prob in detections:
                    if prob > 0.3: # Confidence threshold
                        texts.append(text)
                
                if texts:
                    results.append({
                        "timestamp": timestamp,
                        "text": texts
                    })
            
            frame_idx += 1
            
        cap.release()
        return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <video_path>")
    else:
        extractor = OCRExtractor(gpu=False) # Default to False for safety in testing
        print(extractor.extract(sys.argv[1]))
