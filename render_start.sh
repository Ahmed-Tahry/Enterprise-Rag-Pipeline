#!/bin/bash
# Start FastAPI in background
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 3

# Start Streamlit (foreground)
streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --theme.base=dark
