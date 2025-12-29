
import json
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

class VideoIndexer:
    def __init__(self, output_dir, model_name='all-MiniLM-L6-v2'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading embedding model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS Index
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (Cosine similarity if normalized)
        
        # Metadata storage (list of dicts, parallel to index)
        self.metadata_store = []

    def ingest_metadata(self, metadata_path):
        """
        Loads metadata.json and indexes all text components.
        """
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        video_name = data.get("video_name", "unknown")
        video_path = data.get("video_path", "")
        
        texts_to_encode = []
        temp_meta = []
        
        # 1. Index ASR
        if "asr" in data and data["asr"]:
            for segment in data["asr"]:
                text = segment["text"]
                if len(text.strip()) > 0:
                    texts_to_encode.append(text)
                    temp_meta.append({
                        "source": "ASR",
                        "text": text,
                        "timestamp": segment["start"],
                        "end": segment["end"],
                        "video_name": video_name
                    })
        
        # 2. Index OCR
        if "ocr" in data and data["ocr"]:
            for item in data["ocr"]:
                # OCR returns a list of texts for a frame
                joined_text = " ".join(item["text"])
                if len(joined_text.strip()) > 0:
                    texts_to_encode.append(joined_text)
                    temp_meta.append({
                        "source": "OCR",
                        "text": joined_text,
                        "timestamp": item["timestamp"],
                        "video_name": video_name
                    })
                    
        # 3. Index DET
        if "det" in data and data["det"]:
            for item in data["det"]:
                # Create a description: "Detected person, car, dog."
                objects = [obj["class"] for obj in item["objects"]]
                desc = "Detected objects: " + ", ".join(objects)
                if len(objects) > 0:
                    texts_to_encode.append(desc)
                    temp_meta.append({
                        "source": "DET",
                        "text": desc,
                        "timestamp": item["timestamp"],
                        "video_name": video_name,
                        "objects": objects
                    })

        if not texts_to_encode:
            print("No text found to index.")
            return

        print(f"Encoding {len(texts_to_encode)} segments...")
        embeddings = self.encoder.encode(texts_to_encode)
        
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata_store.extend(temp_meta)
        
        print(f"Index now contains {self.index.ntotal} vectors.")

    def search(self, query, top_k=5):
        print(f"Searching for: '{query}'")
        query_vec = self.encoder.encode([query])
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                meta = self.metadata_store[idx]
                results.append({
                    "score": float(distances[0][i]),
                    "metadata": meta
                })
        return results

    def save(self):
        index_path = self.output_dir / "faiss.index"
        meta_path = self.output_dir / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
        print(f"Index saved to {self.output_dir}")

    def load(self):
        index_path = self.output_dir / "faiss.index"
        meta_path = self.output_dir / "metadata.pkl"
        
        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.metadata_store = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors.")
        else:
            print("No existing index found.")

if __name__ == "__main__":
    # Test
    indexer = VideoIndexer("db/index")
    # Simulate loading a metadata file if one existed
    pass
