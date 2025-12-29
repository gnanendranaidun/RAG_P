
import argparse
import sys
import os

# Wrapper around pipeline.py
try:
    from video_rag.pipeline import VideoRAGPipeline
except ImportError:
    from pipeline import VideoRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Video-RAG Inference")
    parser.add_argument("--db", required=True, help="Path to database directory")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--mode", default="text", choices=["text", "video_llava"], help="Inference mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Error: Database directory '{args.db}' not found. Please run ingest_video.py first.")
        sys.exit(1)
        
    pipeline = VideoRAGPipeline(args.db, args.mode)
    try:
        response = pipeline.run(args.query)
        print("\n>>> ANSWER: " + response)
    except Exception as e:
        print(f"\nERROR during inference: {e}")

if __name__ == "__main__":
    main()
