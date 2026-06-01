import os
import streamlit as st

# Tải API Key bảo mật từ Streamlit Secrets hoặc Biến môi trường
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "AIzaSyBppF0OZ3VPmJ0rhrFAqiYpPYID3YXaiwk"

MODEL_NAME = "models/gemini-3.1-flash-lite"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"