import os
import json
import re
import sys
import time
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Thêm thư mục hiện tại vào sys.path để import các module của dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import GEMINI_API_KEY, MODEL_NAME
from rag_pipeline import create_vector_db, retrieve_docs, generate_answer
from pdf_utils import Document

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo client Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

class EvalScore(BaseModel):
    correctness: int = Field(description="Score from 1 to 5 for correctness against expected answer")
    correctness_reason: str = Field(description="Brief explanation of the correctness score")
    faithfulness: int = Field(description="Score from 1 to 5 for faithfulness against retrieved context")
    faithfulness_reason: str = Field(description="Brief explanation of the faithfulness score")

def load_sample_docs():
    """
    Đọc file extracted_utf8.txt và chuyển đổi thành danh sách các LangChain Documents.
    Đây là nguồn tài liệu dự phòng nếu chưa xây dựng Vector DB thực tế.
    """
    txt_path = os.path.join(os.path.dirname(__file__), "..", "extracted_utf8.txt")
    if not os.path.exists(txt_path):
        logger.error(f"Không tìm thấy file tài liệu mẫu tại {txt_path}")
        return []
        
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    pages = re.split(r'===\s*PAGE\s+(\d+)\s*===', content)
    docs = []
    for i in range(1, len(pages), 2):
        page_num = int(pages[i])
        page_text = pages[i+1].strip()
        if page_text:
            docs.append(Document(
                page_content=page_text,
                metadata={"source": "text", "page": page_num}
            ))
    return docs

def load_ground_truth():
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def judge_answer(query, generated_answer, expected_answer, context):
    """
    Sử dụng Gemini làm trọng tài (LLM-as-a-judge) để đánh giá độ chính xác và tính trung thực.
    """
    prompt = f"""
    Bạn là một chuyên gia đánh giá hệ thống RAG (Retrieval-Augmented Generation). 
    Hãy đánh giá câu trả lời được sinh ra (Generated Answer) dựa trên câu hỏi (Query), ngữ cảnh truy xuất (Context), và câu trả lời mong đợi (Expected Answer).

    CÂU HỎI: {query}
    NGỮ CẢNH TRUY XUẤT: {context}
    CÂU TRẢ LỜI MONG ĐỢI: {expected_answer}
    CÂU TRẢ LỜI ĐƯỢC SINH RA: {generated_answer}

    TIÊU CHÍ ĐÁNH GIÁ:
    1. Answer Correctness (Độ chính xác - thang điểm 1-5): So sánh câu trả lời được sinh ra với câu trả lời mong đợi.
       - 5: Hoàn hảo, trả lời chính xác, đầy đủ sự thật, không thừa thiếu thông tin quan trọng.
       - 4: Tốt, trả lời đúng ý chính nhưng diễn đạt khác hoặc thiếu một vài chi tiết nhỏ không quan trọng.
       - 3: Trung bình, trả lời được một phần thông tin đúng, nhưng có một phần thông tin bị thiếu hoặc sai lệch nhẹ.
       - 2: Kém, hầu hết thông tin là sai lệch hoặc không liên quan đến câu trả lời mong đợi.
       - 1: Hoàn toàn sai hoặc không trả lời được.
       
    2. Faithfulness (Tính trung thực/Không bịa đặt - thang điểm 1-5): Kiểm tra xem câu trả lời được sinh ra có hoàn toàn dựa trên Ngữ cảnh truy xuất hay tự bịa đặt kiến thức bên ngoài.
       - 5: Hoàn toàn trung thực, tất cả các khẳng định trong câu trả lời đều có thể kiểm chứng trực tiếp từ Ngữ cảnh.
       - 4: Khá trung thực, chỉ có ý phụ nhỏ không có trong ngữ cảnh nhưng không ảnh hưởng tới kết quả chính.
       - 3: Trung bình, có một số khẳng định quan trọng không tìm thấy chứng cứ trong ngữ cảnh.
       - 2: Thiếu trung thực, chứa nhiều thông tin bịa đặt không có trong ngữ cảnh.
       - 1: Hoàn toàn bịa đặt (hallucination), không liên quan đến ngữ cảnh.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvalScore,
                temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Lỗi khi chấm điểm LLM-as-a-judge: {e}")
        return {
            "correctness": 1,
            "correctness_reason": f"Lỗi gọi API: {str(e)}",
            "faithfulness": 1,
            "faithfulness_reason": f"Lỗi gọi API: {str(e)}"
        }

def run_evaluation():
    logger.info("=== BẮT ĐẦU ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG RAG ===")
    
    # 1. Khởi tạo Vector DB
    # Kiểm tra xem có faiss_index cục bộ hay không, nếu không thì tự tạo từ file extracted_utf8.txt
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    db = None
    faiss_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
    
    if os.path.exists(faiss_path):
        logger.info(f"Đang tải Vector DB FAISS từ ổ đĩa tại: {faiss_path}")
        try:
            embeddings_model = HuggingFaceEmbeddings()
            db = FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Lỗi tải DB cục bộ: {e}. Sẽ chuyển sang tạo DB tạm thời từ file văn bản mẫu.")
            
    if db is None:
        logger.info("Không tìm thấy DB trên ổ đĩa. Đang tạo Vector DB tạm thời từ extracted_utf8.txt...")
        sample_docs = load_sample_docs()
        if not sample_docs:
            logger.error("Không thể tiếp tục đánh giá vì thiếu tài liệu nguồn.")
            return
        db = create_vector_db(sample_docs)
        
    # 2. Đọc Ground Truth
    ground_truth = load_ground_truth()
    results = []
    
    total_correctness = 0
    total_faithfulness = 0
    total_recall = 0
    total_tests = len(ground_truth)
    
    for item in ground_truth:
        query = item["query"]
        expected_answer = item["expected_answer"]
        target_pages = item["target_pages"]
        
        logger.info(f"Đang chạy câu hỏi ID {item['id']}: {query}")
        
        # Truy xuất tài liệu
        retrieved_docs = retrieve_docs(query, db)
        
        # Tính toán Retrieval Recall
        retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs if doc.metadata.get("page") is not None]
        logger.info(f"Các trang truy xuất được: {retrieved_pages}")
        
        # Recall: Có ít nhất một trang mục tiêu nằm trong danh sách các trang truy xuất được
        recall = 1.0 if any(p in retrieved_pages for p in target_pages) else 0.0
        total_recall += recall
        
        # Ghép nội dung context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Sinh câu trả lời
        generated_answer = generate_answer(query, retrieved_docs, model_name=MODEL_NAME)
        
        # Gọi LLM-as-a-judge chấm điểm
        scores = judge_answer(query, generated_answer, expected_answer, context)
        
        total_correctness += scores["correctness"]
        total_faithfulness += scores["faithfulness"]
        
        results.append({
            "id": item["id"],
            "query": query,
            "expected": expected_answer,
            "generated": generated_answer,
            "retrieved_pages": retrieved_pages,
            "recall": recall,
            "correctness": scores["correctness"],
            "correctness_reason": scores["correctness_reason"],
            "faithfulness": scores["faithfulness"],
            "faithfulness_reason": scores["faithfulness_reason"]
        })
        
        # Chờ 1 giây để tránh chạm rate limit
        time.sleep(1)

    # 3. Tính toán điểm trung bình
    avg_correctness = total_correctness / total_tests
    avg_faithfulness = total_faithfulness / total_tests
    avg_recall = total_recall / total_tests
    
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ===")
    logger.info(f"Tổng số câu hỏi đánh giá: {total_tests}")
    logger.info(f"Độ phủ truy xuất (Retrieval Recall): {avg_recall * 100:.2f}%")
    logger.info(f"Điểm chính xác trung bình (Answer Correctness): {avg_correctness:.2f} / 5.0")
    logger.info(f"Điểm trung thực trung bình (Faithfulness): {avg_faithfulness:.2f} / 5.0")
    
    # 4. Xuất báo cáo Markdown
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đánh Giá Định Lượng Hệ Thống RAG\n\n")
        f.write(f"Ngày đánh giá: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Chỉ số hiệu năng tổng quan\n\n")
        f.write("| Chỉ số | Kết quả | Thang điểm / Mô tả |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| **Retrieval Recall** | **{avg_recall * 100:.1f}%** | Đo lường tỷ lệ tìm thấy đúng trang chứa thông tin mục tiêu |\n")
        f.write(f"| **Answer Correctness** | **{avg_correctness:.2f} / 5.0** | So sánh nội dung câu trả lời của AI với đáp án tham chiếu |\n")
        f.write(f"| **Faithfulness** | **{avg_faithfulness:.2f} / 5.0** | Đo độ trung thực chống bịa đặt (hallucination) dựa trên ngữ cảnh |\n\n")
        
        f.write("## 2. Chi tiết kết quả từng câu hỏi\n\n")
        for r in results:
            f.write(f"### Câu hỏi {r['id']}: {r['query']}\n")
            f.write(f"- **Đáp án tham chiếu:** {r['expected']}\n")
            f.write(f"- **AI trả lời:** {r['generated']}\n")
            f.write(f"- **Các trang truy xuất:** Trang {r['retrieved_pages']} (Recall: {int(r['recall'])})\n")
            f.write(f"- **Điểm Correctness:** {r['correctness']}/5. _Lý do: {r['correctness_reason']}_\n")
            f.write(f"- **Điểm Faithfulness:** {r['faithfulness']}/5. _Lý do: {r['faithfulness_reason']}_\n\n")
            f.write("---\n\n")
            
    logger.info(f"Đã xuất báo cáo đánh giá thành công tại: {report_path}")

if __name__ == "__main__":
    run_evaluation()
