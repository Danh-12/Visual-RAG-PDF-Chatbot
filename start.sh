#!/bin/bash

# Start FastAPI backend in the background
echo "🟢 Starting FastAPI Backend on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for uvicorn to bind
sleep 3

# Start Streamlit frontend in the foreground
echo "🟢 Starting Streamlit Frontend on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
