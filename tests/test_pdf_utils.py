import sys
import os
from unittest.mock import MagicMock

# Thêm thư mục gốc vào sys.path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdf_utils import extract_documents, extract_images

def test_extract_documents():
    # Mock fitz pages
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Nội dung trang 1"
    
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "   " # Trang trắng
    
    mock_page3 = MagicMock()
    mock_page3.get_text.return_value = "Nội dung trang 3"
    
    mock_doc = [mock_page1, mock_page2, mock_page3]
    
    docs = extract_documents(mock_doc)
    
    assert len(docs) == 2
    assert docs[0].page_content == "Nội dung trang 1"
    assert docs[0].metadata["page"] == 1
    assert docs[1].page_content == "Nội dung trang 3"
    assert docs[1].metadata["page"] == 3

def test_extract_images_filtered_small():
    # Trả về ảnh có kích thước nhỏ (100x100), sẽ bị lọc bỏ
    mock_page = MagicMock()
    mock_page.get_images.return_value = [(10, 0, 100, 100, 8, "DeviceRGB", "", "img1", "DCTDecode")]
    
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.extract_image.return_value = {
        "image": b"dummy_bytes_short",
        "ext": "png",
        "width": 100,
        "height": 100
    }
    
    images = extract_images(mock_doc)
    assert len(images) == 0

def test_extract_images_large_diagram():
    # Trả về ảnh có kích thước lớn (500x400) và dung lượng lớn, sẽ được giữ lại
    mock_page = MagicMock()
    mock_page.get_images.return_value = [(20, 0, 500, 400, 8, "DeviceRGB", "", "img2", "DCTDecode")]
    
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.extract_image.return_value = {
        "image": b"dummy_bytes_large" * 500,
        "ext": "png",
        "width": 500,
        "height": 400
    }
    
    images = extract_images(mock_doc)
    assert len(images) == 1
    assert images[0]["page"] == 1
    assert images[0]["xref"] == 20
    assert images[0]["ext"] == "png"
