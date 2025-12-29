
import cv2
import numpy as np

def create_video(output_path="test_video.mp4"):
    width, height = 640, 480
    fps = 30
    seconds = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print(f"Generating {output_path}...")
    
    for i in range(fps * seconds):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background color change
        frame[:] = (50, 50, 50) 
        
        # Add Text (for OCR)
        if i < 60:
            cv2.putText(frame, 'Hello World', (50, 200), font, 2, (255, 255, 255), 3)
        elif i < 120:
            cv2.putText(frame, 'Video RAG Test', (50, 200), font, 2, (255, 255, 255), 3)
            
        # Add Object (Red Circle for DET)
        cv2.circle(frame, (320, 240), 50, (0, 0, 255), -1)
        
        out.write(frame)
        
    out.release()
    print("Video generated.")

if __name__ == "__main__":
    create_video()
