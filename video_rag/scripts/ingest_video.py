
import os
import json
import argparse
from pathlib import Path
import torch

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.extraction.asr import ASRExtractor
from src.extraction.ocr import OCRExtractor
from src.extraction.det import DetExtractor

def ingest_video(video_path, output_dir, device=None):
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem
    output_dir = Path(output_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    metadata = {
        "video_path": video_path,
        "video_name": video_name
    }
    
    # 1. ASR
    asr = ASRExtractor(device=device)
    metadata["asr"] = asr.extract(video_path)
    
    # 2. OCR
    # OCR uses CPU or GPU inside the class, passed via init if needed. 
    # For now ensuring we pass gpu flag correctly.
    use_gpu = torch.cuda.is_available() or (torch.backends.mps.is_available())
    ocr = OCRExtractor(gpu=use_gpu) 
    metadata["ocr"] = ocr.extract(video_path, interval_seconds=1)
    
    # 3. Object Detection
    det = DetExtractor()
    metadata["det"] = det.extract(video_path, interval_seconds=1)
    
    # Save metadata
    output_file = output_dir / "metadata.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Ingestion complete. Metadata saved to {output_file}")
    
    # 4. Initialize Indexer and Index
    from src.retrieval.indexer import VideoIndexer
    indexer = VideoIndexer(output_dir)
    indexer.ingest_metadata(output_file)
    indexer.save()
    print("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest video for Video-RAG")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="db", help="Output directory for database")
    parser.add_argument("--device", default=None, help="Device for models (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    ingest_video(args.video, args.output, args.device)
