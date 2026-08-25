import os
import shutil
import base64
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODEL_NAME
from pdf_utils import extract_documents, extract_images
from rag_pipeline import (
    split_documents, 
    generate_image_description, 
    create_vector_db, 
    save_vector_db, 
    load_vector_db,
    retrieve_docs, 
    generate_answer,
    Document
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Visual RAG API Backend",
    description="REST endpoints for Multimodal RAG with FAISS persistence and Gemini LLM",
    version="1.0.0"
)

# Thêm CORS middleware để cho phép các domain khác kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FOLDER = "faiss_index"
STATS_FILE = "faiss_index/stats.json"

class QueryRequest(BaseModel):
    query: str
    model_name: Optional[str] = MODEL_NAME
    chat_history: Optional[List[dict]] = []

class DocumentMetadata(BaseModel):
    source: str
    page: int
    image_base64: Optional[str] = None
    ext: Optional[str] = None

class DocumentResponse(BaseModel):
    page_content: str
    metadata: DocumentMetadata

class QueryResponse(BaseModel):
    answer: str
    docs: List[DocumentResponse]

@app.get("/api/stats")
async def get_stats():
    """
    Trả về thống kê trạng thái hiện tại của Vector DB.
    """
    if not os.path.exists(DB_FOLDER):
        return {"ready": False, "pages": 0, "chunks": 0, "images": 0}
    
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
                return {"ready": True, **stats}
        except Exception:
            pass
            
    return {"ready": True, "pages": "Không rõ", "chunks": "Không rõ", "images": "Không rõ"}

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    analyze_images: bool = Form(True),
    max_images: int = Form(30),
    active_model: str = Form(MODEL_NAME)
):
    """
    Nhận file PDF, phân tách text, phân tích ảnh bằng Vision AI, 
    xây dựng Vector DB FAISS và lưu xuống đĩa.
    """
    logger.info(f"Nhận yêu cầu xử lý file PDF: {file.filename}")
    temp_pdf_path = f"temp_{file.filename}"
    
    try:
        # Lưu file tạm thời
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Đọc PDF bằng PyMuPDF
        doc = fitz.open(temp_pdf_path)
        logger.info(f"Đã mở PDF: {len(doc)} trang")
        
        # Trích xuất văn bản
        text_docs = extract_documents(doc)
        text_chunks = split_documents(text_docs)
        logger.info(f"Trích xuất thành công {len(text_chunks)} phân đoạn văn bản.")
        
        # 2. Trích xuất hình ảnh
        image_docs = []
        images = []
        
        if analyze_images:
            images = extract_images(doc)
            logger.info(f"Phát hiện {len(images)} hình ảnh/sơ đồ kỹ thuật.")
            
            if images:
                actual_max = min(len(images), max_images)
                logger.info(f"Sẽ tiến hành phân tích tối đa {actual_max} ảnh sử dụng mô hình Vision.")
                
                # Định nghĩa hàm xử lý ảnh
                def process_image(idx, img):
                    should_describe = idx < max_images
                    desc = ""
                    if should_describe:
                        try:
                            desc = generate_image_description(img["image_bytes"], img["ext"], model_name=active_model)
                            desc = desc.strip()
                        except Exception as e:
                            logger.error(f"Lỗi Vision AI cho ảnh tại trang {img['page']}: {e}")
                    
                    if not desc:
                        if should_describe:
                            desc = (f"Sơ đồ thực tế tại trang {img['page']}. "
                                    f"(Ghi chú: Bản mô tả chi tiết bằng AI chưa được nạp do lỗi quota hoặc lỗi kết nối).")
                        else:
                            desc = f"Sơ đồ thực tế tại trang {img['page']}."
                            
                    return Document(
                        page_content=f"[Sơ đồ/Bảng biểu ở trang {img['page']}]:\n{desc}",
                        metadata={
                            "source": "image",
                            "page": img["page"],
                            "image_bytes": img["image_bytes"],
                            "ext": img["ext"]
                        }
                    )
                
                # Chạy song song luồng xử lý ảnh
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(process_image, idx, img) for idx, img in enumerate(images)]
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            if res:
                                image_docs.append(res)
                        except Exception as e:
                            logger.error(f"Lỗi xử lý luồng hình ảnh: {e}")
                            
        # 3. Kết hợp tài liệu và tạo Vector DB FAISS
        all_chunks = text_chunks + image_docs
        logger.info(f"Tổng hợp {len(all_chunks)} chunks văn bản + ảnh để index.")
        
        db = create_vector_db(all_chunks)
        
        # Lưu Vector DB xuống đĩa
        if os.path.exists(DB_FOLDER):
            shutil.rmtree(DB_FOLDER)
        os.makedirs(DB_FOLDER, exist_ok=True)
        save_vector_db(db, DB_FOLDER)
        
        # Lưu stats
        stats_data = {
            "pages": len(doc),
            "chunks": len(text_chunks),
            "images": len(images)
        }
        
        import json
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_data, f)
            
        logger.info("Đã xây dựng và lưu Vector DB FAISS thành công.")
        return {"status": "success", "stats": stats_data}
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình upload & index PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Xóa file PDF tạm
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Nhận câu hỏi, tìm kiếm ngữ cảnh từ FAISS DB đã lưu, sinh câu trả lời và trả về.
    """
    if not os.path.exists(DB_FOLDER):
        raise HTTPException(status_code=400, detail="Chưa có tài liệu nào được nạp vào Vector DB.")
        
    try:
        # Load Vector DB từ đĩa
        db = load_vector_db(DB_FOLDER)
        
        # Tìm kiếm ngữ cảnh
        docs = retrieve_docs(request.query, db)
        
        # Sinh câu trả lời
        answer = generate_answer(
            query=request.query,
            docs=docs,
            model_name=request.model_name,
            chat_history=request.chat_history
        )
        
        # Serialize các tài liệu tìm được kèm mã hóa base64 cho ảnh
        serialized_docs = []
        for doc in docs:
            image_base64 = None
            if "image_bytes" in doc.metadata and doc.metadata["image_bytes"]:
                image_base64 = base64.b64encode(doc.metadata["image_bytes"]).decode("utf-8")
                
            serialized_docs.append({
                "page_content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "text"),
                    "page": doc.metadata.get("page", 0),
                    "image_base64": image_base64,
                    "ext": doc.metadata.get("ext")
                }
            })
            
        return {
            "answer": answer,
            "docs": serialized_docs
        }
        
    except Exception as e:
        logger.error(f"Lỗi truy vấn RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
