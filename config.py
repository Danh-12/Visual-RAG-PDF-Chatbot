import os
import streamlit as st
from dotenv import load_dotenv

# Tải file .env nếu có
load_dotenv()

# Tải API Key bảo mật từ Streamlit Secrets hoặc Biến môi trường
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

MODEL_NAME = "models/gemini-3.1-flash-lite"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"
API_URL = st.secrets.get("API_URL") or os.environ.get("API_URL") or os.getenv("API_URL") or "http://localhost:8000"