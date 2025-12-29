
import argparse
import os
import sys
import torch
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
        
        # 2. Generate
        answer = self.llm.generate(query, context_text=combined_context)
        
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
