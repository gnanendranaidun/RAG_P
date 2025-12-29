
import argparse
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Add directory containing this file to path to allow imports from src
sys.path.append(os.path.dirname(__file__))

from src.retrieval.indexer import VideoIndexer
from src.model.llm import LLMInterface

class VideoRAGPipeline:
    def __init__(self, db_path, model_mode="text"):
        self.indexer = VideoIndexer(db_path)
        self.indexer.load()
        self.llm = LLMInterface(mode=model_mode)

    def extract_frame(self, video_path, timestamp):
        """Extracts a frame from the video at the given timestamp."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_no = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
        except Exception as e:
            print(f"Error extracting frame: {e}")
        return None

    def run(self, query):
        print(f"Querying: {query}")
        
        # 1. Retrieve
        results = self.indexer.search(query, top_k=5)
        
        print("\n--- Retrieved Contexts ---")
        context_parts = []
        for res in results:
            meta = res["metadata"]
            score = res["score"]
            text = f"[{meta['source']} @ {meta.get('timestamp', 0):.2f}s]: {meta['text']}"
            context_parts.append(text)
            print(f"({score:.4f}) {text}")
        print("--------------------------\n")
        
        combined_context = "\n".join(context_parts)
        
        # 2. Extract Frame from Top Result if available
        image_input = None
        if results:
            top_meta = results[0]["metadata"]
            if "video_path" in top_meta:
                print(f"Extracting frame from {top_meta['video_name']} at {top_meta.get('timestamp', 0):.2f}s...")
                image_input = self.extract_frame(top_meta["video_path"], top_meta.get("timestamp", 0))
        
        # 3. Generate
        answer = self.llm.generate(query, context_text=combined_context, image_input=image_input)
        
        return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to database directory containing index")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--mode", default="text", help="LLM mode: text or video_llava")
    
    args = parser.parse_args()
    
    pipeline = VideoRAGPipeline(args.db, args.mode)
    response = pipeline.run(args.query)
    
    print("\n=== FINAL ANSWER ===")
    print(response)
    print("====================")
