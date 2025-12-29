
from ultralytics import YOLO
import cv2

class DetExtractor:
    def __init__(self, model_name="yolov8n.pt"):
        print(f"Loading YOLO model '{model_name}'...")
        self.model = YOLO(model_name)

    def extract(self, video_path, interval_seconds=1):
        """
        Extracts objects from video frames sampled at interval_seconds.
        """
        print(f"Extracting objects from {video_path}...")
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
                
                # Run inference
                prediction = self.model(frame, verbose=False)[0]
                
                objects = []
                for box in prediction.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    if conf > 0.3:
                        objects.append({
                            "class": class_name,
                            "confidence": conf,
                            "box": box.xyxy[0].tolist()
                        })
                
                if objects:
                    # Count objects per class
                    counts = {}
                    for obj in objects:
                        cls = obj["class"]
                        counts[cls] = counts.get(cls, 0) + 1
                        
                    results.append({
                        "timestamp": timestamp,
                        "objects": objects, # Keep detailed objects if needed later
                        "counts": counts
                    })
            
            frame_idx += 1
            
        cap.release()
        return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python det.py <video_path>")
    else:
        extractor = DetExtractor()
        print(extractor.extract(sys.argv[1]))
