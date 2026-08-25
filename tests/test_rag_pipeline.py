import sys
import os
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Thêm thư mục gốc vào sys.path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_pipeline import split_documents, create_vector_db, generate_answer

def test_split_documents():
    docs = [Document(page_content="Nội dung kiểm thử " * 200, metadata={"page": 1})]
    split_docs = split_documents(docs)
    assert len(split_docs) > 1
    assert all(d.metadata["page"] == 1 for d in split_docs)

@patch("langchain_community.embeddings.HuggingFaceEmbeddings")
@patch("langchain_community.vectorstores.FAISS")
def test_create_vector_db(mock_faiss, mock_embeddings):
    docs = [Document(page_content="Nội dung test database", metadata={"page": 1})]
    mock_db = MagicMock()
    mock_faiss.from_documents.return_value = mock_db
    
    db = create_vector_db(docs)
    
    mock_embeddings.assert_called_once()
    mock_faiss.from_documents.assert_called_once_with(docs, mock_embeddings.return_value)
    assert db == mock_db

@patch("rag_pipeline.client")
def test_generate_answer(mock_client):
    # Mock API call của Gemini để không tốn quota khi chạy unit test
    mock_response = MagicMock()
    mock_response.text = "Mô hình trả lời: Hệ thống đặt lịch trực tuyến."
    mock_client.models.generate_content.return_value = mock_response
    
    docs = [Document(page_content="Đề tài là đặt lịch trực tuyến.", metadata={"page": 1, "source": "text"})]
    answer = generate_answer("Đề tài là gì?", docs, model_name="models/dummy-model")
    
    assert answer == "Mô hình trả lời: Hệ thống đặt lịch trực tuyến."
    mock_client.models.generate_content.assert_called_once()
