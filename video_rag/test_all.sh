
#!/bin/bash
set -e

source .venv/bin/activate

echo ">>> Step 1: generating dummy video..."
python scripts/create_dummy_video.py

echo ">>> Step 2: Ingesting video..."
python scripts/ingest_video.py --video test_video.mp4 --output db

echo ">>> Step 3: Running Inference..."
python run_inference.py --db db/test_video --query "What text is shown on the video?"

echo ">>> Verification Complete!"
