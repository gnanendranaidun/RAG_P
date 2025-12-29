import os
import cv2
import numpy as np
import torch
import faiss
import whisper
import easyocr
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
VIDEO_PATH = "test_video.mp4"
DB_DIR = "db/video_rag_db"
os.makedirs(DB_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 1. Video Segmentation
def extract_frames(video_path, fps=1.0):
    frames = []
    timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        
        count += 1
    
    cap.release()
    return frames, timestamps

print("Extracting frames...")
frames, timestamps = extract_frames(VIDEO_PATH, fps=1.0)
print(f"Extracted {len(frames)} frames from {VIDEO_PATH}")

# 2. ASR
print("Transcribing audio...")
# Use 'base' or 'small' for speed if needed, 'turbo' if available, paper suggests just 'Whisper'
asr_model = whisper.load_model("base", device=DEVICE)
result = asr_model.transcribe(VIDEO_PATH)
segments = result['segments']

asr_docs = []
for seg in segments:
    text = seg['text'].strip()
    if text:
        asr_docs.append({
            "text": text,
            "start": seg['start'],
            "end": seg['end'],
            "type": "ASR"
        })
print(f"Generated {len(asr_docs)} ASR segments.")

# 3. OCR
print("Running OCR...")
# EasyOCR on MPS might have issues or fallback to CPU if not supported well, but try GPU arg
reader = easyocr.Reader(['en'], gpu=(DEVICE != 'cpu'))
ocr_docs = []

for i, frame in enumerate(tqdm(frames)):
    try:
        results = reader.readtext(frame)
        frame_text = " ".join([res[1] for res in results])
        if frame_text.strip():
            ocr_docs.append({
                "text": frame_text,
                "timestamp": timestamps[i],
                "frame_idx": i,
                "type": "OCR"
            })
    except Exception as e:
        print(f"OCR error on frame {i}: {e}")

print(f"Found text in {len(ocr_docs)} frames.")

# 4. Indexing
print("Building Indices...")
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)

def build_faiss_index(docs, index_name):
    if not docs:
        print(f"Skipping {index_name}, no docs.")
        return
    
    texts = [d['text'] for d in docs]
    embeddings = retriever.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(DB_DIR, f"{index_name}.index"))
    with open(os.path.join(DB_DIR, f"{index_name}_meta.pkl"), "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved {index_name} index.")

build_faiss_index(asr_docs, "asr")
build_faiss_index(ocr_docs, "ocr")

# 5. Visual Features
print("Extracting Visual Embeddings...")
clip_model_name = "openai/clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)

batch_size = 16 # Reduce batch size for safety
visual_embeddings = []

with torch.no_grad():
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        # inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        # Check if we need to convert numpy to list of PIL or if processor handles it
        # Processor handles list of numpy arrays (H,W,C) usually.
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
        
        outputs = clip_model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        visual_embeddings.append(outputs.cpu())

if visual_embeddings:
    visual_features = torch.cat(visual_embeddings)
    torch.save(visual_features, os.path.join(DB_DIR, "visual_embeddings.pt"))
    print(f"Saved visual embeddings: {visual_features.shape}")

print("Phase 1 Complete.")
