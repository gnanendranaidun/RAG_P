# Video RAG Inference Architecture

This diagram illustrates the process when you run the inference command.

```mermaid
sequenceDiagram
    participant User
    participant CLI as run_inference.py
    participant Pipeline as VideoRAGPipeline
    participant Indexer as VideoIndexer (FAISS)
    participant Model as LLMInterface (DistilGPT2)

    User->>CLI: python run_inference.py --query "..."
    
    rect rgb(240, 240, 240)
    Note over CLI, Model: 1. Initialization Phase
    CLI->>Pipeline: Initialize(db_path, mode)
    Pipeline->>Indexer: load()
    Indexer-->>Pipeline: Load faiss.index & metadata.pkl
    Pipeline->>Model: Initialize(text_model)
    Model-->>Pipeline: Load Model & Tokenizer (on MPS)
    end

    rect rgb(230, 240, 255)
    Note over CLI, Model: 2. Execution Phase
    CLI->>Pipeline: run(query)
    
    Note right of Pipeline: Retrieval
    Pipeline->>Indexer: search(query, top_k=5)
    Indexer->>Indexer: Encode Query (SentenceTransformer)
    Indexer->>Indexer: Vector Search (FAISS)
    Indexer-->>Pipeline: Return Top Relevant Contexts (ASR/OCR)
    
    Note right of Pipeline: Generation
    Pipeline->>Pipeline: Create Prompt (Context + Question)
    Pipeline->>Model: generate(prompt)
    Model->>Model: Tokenize -> Inference -> Decode
    Model-->>Pipeline: Return Answer Text
    end

    Pipeline-->>CLI: Final Answer
    CLI-->>User: Print Output
```

## Detailed Steps

1.  **Loading**:
    *   The script loads the **FAISS Index** (vectors) and **Metadata** (text segments) from your disk (`db/christmas_run/test_video`).
    *   It loads the **LLM** (`distilgpt2`) into your Mac's GPU memory (MPS).

2.  **Retrieval**:
    *   Your question is converted into a vector.
    *   The system searches for the most similar vectors in the index.
    *   It finds the relevant segment: *"[ASR @ 33.04s]: Small evergreen trees were decorated..."*

3.  **Generation**:
    *   The system constructs a prompt:
        ```text
        Context:
        [ASR @ 33.04s]: Small evergreen trees were decorated...
        
        Question: how many berry, apple and candles...
        Answer:
        ```
    *   The LLM reads this prompt and completes the text to generate the answer.
