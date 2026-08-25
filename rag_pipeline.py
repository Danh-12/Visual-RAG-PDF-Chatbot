import sys
import time
import logging

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, MODEL_NAME, EMBEDDING_MODEL_NAME
# ===== INIT GEMINI =====
client = genai.Client(api_key=GEMINI_API_KEY)
# ===== CUSTOM GEMINI EMBEDDINGS =====
class GeminiEmbeddings(Embeddings):
    def __init__(self, client, model_name=EMBEDDING_MODEL_NAME):
        self.client = client
        self.model_name = model_name
        self.dim = None
        # Proactively fetch dimension to ensure self._zero_vector matches the index dimension
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents="test"
            )
            self.dim = len(response.embeddings[0].values)
            logger.info("Detected embedding dimension: %s", self.dim)
        except Exception as e:
            logger.error("Failed to fetch embedding dimension: %s", e)
            self.dim = 3072 # default fallback dimension for models/gemini-embedding-001 in google-genai

    def _zero_vector(self):
        return [0.0] * (self.dim or 3072)

    def _normalize_vector(self, values):
        values = list(values)

        if self.dim is None:
            self.dim = len(values)

        if len(values) != self.dim:
            logger.error("Embedding size error: %s != %s", len(values), self.dim)
            return self._zero_vector()

        return values

    def embed_documents(self, texts):
        embeddings = []
        batch_size = 80

        for i in range(0, len(texts), batch_size):
            if i > 0:
                time.sleep(1.5)

            batch = texts[i:i + batch_size]
            batch_cleaned = [t if t and t.strip() else " " for t in batch]

            contents_list = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch_cleaned
            ]

            # Retry logic with backoff for rate limits
            response = None
            for attempt in range(5):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=contents_list
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "503" in str(e):
                        sleep_time = 3 * (attempt + 1)
                        logger.warning("Embedding batch rate limited. Retrying in %ss...", sleep_time)
                        time.sleep(sleep_time)
                    else:
                        logger.error("Embedding batch API error: %s", e)
                        break

            if response:
                for emb in response.embeddings:
                    embeddings.append(self._normalize_vector(emb.values))
            else:
                logger.error("Embedding batch failed after retries, using zero vectors.")
                for _ in batch:
                    embeddings.append(self._zero_vector())

        return embeddings

    def embed_query(self, text):
        cleaned_text = text if text and text.strip() else " "

        response = None
        for attempt in range(5):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=[
                        types.Content(parts=[types.Part.from_text(text=cleaned_text)])
                    ]
                )
                break
            except Exception as e:
                if "429" in str(e) or "503" in str(e):
                    sleep_time = 2 * (attempt + 1)
                    logger.warning("Query embedding rate limited. Retrying in %ss...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("Query embedding API error: %s", e)
                    break

        if response:
            return self._normalize_vector(response.embeddings[0].values)
        else:
            return self._zero_vector()
        
# ===== MULTIMODAL IMAGE DESCRIPTION =====

def generate_image_description(image_bytes, image_ext, model_name=None):
    """
    Gửi ảnh thô trích xuất từ PDF qua Gemini 2.5 Flash để mô tả chi tiết nội dung.
    Gửi ảnh thô trích xuất từ PDF qua Gemini để mô tả chi tiết nội dung.
    Điều này cho phép tìm kiếm ngữ nghĩa sau này khớp được với sơ đồ, bảng biểu.
    Có cơ chế tự động thử lại (retry) với exponential backoff nếu gặp Rate Limit (429).
    """
    active_model = model_name if model_name not in [None, ""] else MODEL_NAME
    prompt = """
    Hãy đóng vai trò là một chuyên gia phân tích tài liệu khoa học/kỹ thuật. 
    Nhiệm vụ của bạn là đọc và phân tích hình ảnh/sơ đồ/bảng biểu này và cung cấp mô tả chi tiết:
    1. Thể loại (Bảng số liệu, biểu đồ cột/đường/tròn, sơ đồ khối quy trình, hình vẽ kỹ thuật, ảnh chụp minh họa...).
    2. Nội dung chi tiết: đọc và liệt kê chính xác các con số, tiêu đề bảng, tên cột/dòng, nhãn của các trục, các khối chức năng và luồng kết nối trong sơ đồ.
    3. Ý nghĩa hoặc kết luận chính từ sơ đồ/bảng biểu này.
    Vui lòng viết mô tả bằng tiếng Việt rõ ràng, mạch lạc, có cấu trúc tốt.
    """

    for attempt in range(5):
        try:
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=f"image/{image_ext}"
            )
            response = client.models.generate_content(
                model=active_model,
                contents=[image_part, prompt]
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                if attempt == 4:
                    logger.error("Image analysis error (max attempts exceeded): %s", e)
                    return ""
                # Chờ đợi với exponential backoff: 3s, 6s, 12s, 24s...
                sleep_time = 3 * (attempt + 1)
                logger.warning("API rate limited or connection error. Retrying in %ss...", sleep_time)
                try:
                    import streamlit as st
                    st.toast(f"⚠️ API mô hình bị nghẽn (429/503) khi phân tích hình ảnh. Thử lại sau {sleep_time} giây...", icon="⏳")
                except Exception:
                    pass
                time.sleep(sleep_time)
            else:
                logger.error("Image analysis error by Gemini: %s", e)
                return ""
    return ""

def split_documents(documents):
    """
    Tách tài liệu thành các đoạn nhỏ để đưa vào Vector DB.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
    return splitter.split_documents(documents)

# ===== VECTOR DB =====
def create_vector_db(documents):
    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if not documents:
        raise ValueError("Không có nội dung hợp lệ để tạo Vector DB.")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings_model = HuggingFaceEmbeddings()
    
    db = FAISS.from_documents(documents, embeddings_model)
    return db

def save_vector_db(db, folder_path="faiss_index"):
    """
    Lưu Vector DB FAISS xuống đĩa cục bộ.
    """
    db.save_local(folder_path)

def load_vector_db(folder_path="faiss_index"):
    """
    Tải Vector DB FAISS từ đĩa cục bộ.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    embeddings_model = HuggingFaceEmbeddings()
    db = FAISS.load_local(folder_path, embeddings_model, allow_dangerous_deserialization=True)
    return db

# ===== RETRIEVE =====
def retrieve_docs(query, db):
    # 1. Perform semantic search (increase k to 7 to improve recall)
    docs = db.similarity_search(query, k=7)
    
    # Count total stats and unique captioned figures from docstore dynamically
    total_pages = 0
    total_text_chunks = 0
    total_images = 0
    unique_captions = set()
    try:
        if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
            for doc in db.docstore._dict.values():
                page = doc.metadata.get("page", 0)
                source = doc.metadata.get("source", "text")
                if page > total_pages:
                    total_pages = page
                if source == "image":
                    total_images += 1
                else:
                    total_text_chunks += 1
                    # Quét nhãn chú thích hình ảnh chính thức (Hình X hoặc Hình X.Y)
                    import re
                    matches = re.findall(r'(?:Hình|Sơ đồ)\s+(\d+(?:\.\d+)*)', doc.page_content)
                    for m in matches:
                        unique_captions.add(m)
    except Exception as e:
        logger.error("Error counting dynamic stats: %s", e)

    # Nếu không quét được nhãn chú thích nào bằng chữ, mặc định lấy số lượng ảnh vật lý để tránh hiển thị sai lệch
    total_captioned_figures = len(unique_captions) if unique_captions else total_images

    # Create dynamic system metadata
    metadata_content = (
        f"[Thông tin cấu trúc tổng quan của tài liệu PDF đang phân tích]:\n"
        f"- Tổng số trang: {total_pages}\n"
        f"- Tổng số sơ đồ, hình ảnh có chú thích chính thức trong báo cáo (nhãn Hình X.Y): {total_captioned_figures}\n"
        f"- Tổng số đối tượng ảnh vật lý được trích xuất (bao gồm cả các ảnh con phụ trợ ghép trong hình chính): {total_images}\n"
        f"- Tổng số phân đoạn văn bản: {total_text_chunks}\n"
    )
    system_metadata_doc = Document(
        page_content=metadata_content,
        metadata={"source": "system", "page": 0}
    )
    
    # 2. Extract page numbers mentioned in the query
    import re
    page_numbers = []
    # Match patterns like "trang 33", "trang 37", "page 33", "p.33", etc.
    matches = re.findall(r'(?:trang|page|p\.?)\s*(\d+)', query.lower())
    for m in matches:
        try:
            page_numbers.append(int(m))
        except ValueError:
            pass
            
    # Also find standalone numbers between 1 and 300 (likely pages)
    standalone_numbers = re.findall(r'\b(\d{1,3})\b', query)
    for num_str in standalone_numbers:
        num = int(num_str)
        if 1 <= num <= 300:
            if num not in page_numbers:
                page_numbers.append(num)
                
    # 3. Detect if this is a global image query asking for all figures/diagrams
    query_lower = query.lower()
    # Check for words indicating listing/counting and nouns indicating figures/diagrams
    has_action = any(verb in query_lower for verb in [
        "liệt kê", "danh sách", "tất cả", "toàn bộ", "mọi", "bao nhiêu", "có những",
        "show", "list", "all", "every", "count", "đếm", "những hình", "các hình",
        "sơ đồ có trong", "hình ảnh có trong"
    ])
    has_noun = any(noun in query_lower for noun in [
        "hình", "sơ đồ", "biểu đồ", "ảnh", "diagram", "figure", "image", "chart"
    ])
    is_global_image_query = has_action and has_noun
                
    # 4. Extract cover pages (Pages 1 to 2) and targeted diagrams from the docstore as background context
    global_docs = []
    targeted_image_docs = []
    global_image_docs = []
    try:
        if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
            # Sort documents by page number to maintain logical order
            sorted_docs = sorted(
                db.docstore._dict.values(),
                key=lambda d: d.metadata.get("page", 0)
            )
            for doc in sorted_docs:
                page = doc.metadata.get("page")
                source = doc.metadata.get("source")
                
                # Collect all image docs for global queries
                if source == "image":
                    global_image_docs.append(doc)
                
                # Check for cover pages
                if page and 1 <= page <= 2:
                    if not any(gd.page_content == doc.page_content for gd in global_docs):
                        global_docs.append(doc)
                        
                # Check for targeted documents (both text and image) on mentioned pages
                if page in page_numbers:
                    if not any(td.page_content == doc.page_content for td in targeted_image_docs):
                        targeted_image_docs.append(doc)
    except Exception as e:
        logger.error("Error extracting context pages from docstore: %s", e)
        
    # Combine: System Metadata is ALWAYS first to provide global counts and stats
    final_docs = [system_metadata_doc]
    
    if is_global_image_query:
        # Feed all image descriptions so LLM can list/count them accurately
        for doc in global_image_docs:
            if not any(fd.page_content == doc.page_content for fd in final_docs):
                final_docs.append(doc)
        # Append similarity search results as secondary context
        for doc in docs:
            if not any(fd.page_content == doc.page_content for fd in final_docs):
                final_docs.append(doc)
    else:
        # First, append targeted image documents so they are at the top and guaranteed
        for doc in targeted_image_docs:
            if not any(fd.page_content == doc.page_content for fd in final_docs):
                final_docs.append(doc)
                
        # Then append similarity search results
        for doc in docs:
            if not any(fd.page_content == doc.page_content for fd in final_docs):
                final_docs.append(doc)
                
        # Then append the global cover context at the end
        for doc in global_docs:
            if not any(fd.page_content == doc.page_content for fd in final_docs):
                final_docs.append(doc)
            
    # Limit dynamically based on query type to prevent token exhaustion but allow high recall
    if is_global_image_query:
        final_docs = final_docs[:40]
    else:
        final_docs = final_docs[:10]

    logger.info("QUERY: %s", query)
    logger.info("Is Global Image Query: %s", is_global_image_query)
    logger.info("Detected page numbers: %s", page_numbers)
    logger.info("Original Similarity Chunks: %s", len(docs))
    logger.info("Targeted Image Chunks added: %s", len(targeted_image_docs))
    logger.info("Global Cover Pages (1-2) Chunks added: %s", len(global_docs))
    logger.info("Total Context Chunks: %s", len(final_docs))

    for i, d in enumerate(final_docs):
        logger.info("[%s] Page: %s | Source: %s", i, d.metadata.get("page"), d.metadata.get("source"))
        logger.info(d.page_content[:150].replace('\n', ' '))

    return final_docs
# ===== GENERATE ANSWER WITH CITATIONS & ANTI-HALLUCINATION =====

def generate_answer(query, docs, model_name=None, chat_history=None):
    if not docs:
        return "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
    """
    Tạo câu trả lời từ các đoạn tài liệu truy xuất được, kèm trích dẫn trang rõ ràng.
    Áp dụng prompt engineering nghiêm ngặt để chống hallucination.
    Tích hợp bộ nhớ hội thoại FIFO để hỗ trợ câu hỏi nối tiếp.
    """
    active_model = model_name if model_name not in [None, ""] else MODEL_NAME
    context_parts = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", "không rõ")
        source_type = doc.metadata.get("source", "text")
        
        if source_type == "system":
            source_desc = f"[Tham khảo {i+1} - Thông tin thống kê tổng quan của hệ thống]"
        elif source_type == "image":
            source_desc = f"[Tham khảo {i+1} - Sơ đồ/Bảng biểu tại Trang {page_num}]"
        else:
            source_desc = f"[Tham khảo {i+1} - Văn bản tại Trang {page_num}]"
            
        context_parts.append(f"{source_desc}:\n{doc.page_content}")
        
    context = "\n\n".join(context_parts)
    
    # Cấu hình FIFO cho lịch sử chat (chỉ lấy tối đa 3 câu hỏi-trả lời gần nhất để tối ưu token)
    history_context = ""
    if chat_history:
        fifo_history = chat_history[-3:]
        history_parts = []
        for idx, turn in enumerate(fifo_history):
            history_parts.append(f"Q{idx+1}: {turn['query']}\nA{idx+1}: {turn['answer']}")
        history_context = "\n".join(history_parts)
    
    prompt = f"""
Bạn là một trợ lý AI thông minh chuyên phân tích tài liệu với độ chính xác tuyệt đối. 
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRÊN VÀ CHỈ TRÊN các tài liệu tham khảo (Context) được cung cấp dưới đây.
---
BỐI CẢNH TÀI LIỆU (CONTEXT):
{context}
"""
    if history_context:
        prompt += f"""
---
LỊCH SỬ HỘI THOẠI GẦN NHẤT (FIFO MEMORY):
{history_context}
"""
        
    prompt += f"""
---
CÂU HỎI CỦA NGƯỜI DÙNG: {query}
---
YÊU CẦU NGHIÊM NGẶT ĐỂ TRÁNH TRẢ LỜI SAI LỆCH (CHỐNG BỊA ĐẶT / HALLUCINATION):
1. Bạn CHỈ được trả lời bằng các thông tin có trong phần Context ở trên. TUYỆT ĐỐI không tự suy diễn ngoài lề, không lấy kiến thức bên ngoài hệ thống hoặc bịa đặt thông tin. Lịch sử hội thoại gần nhất chỉ dùng để hiểu ngữ cảnh của câu hỏi tiếp theo (ví dụ: các đại từ xưng hô, câu hỏi nối tiếp như "tại sao?"), không dùng để thay thế tài liệu Context hiện tại.
2. Nếu Context không chứa đủ thông tin để trả lời đầy đủ câu hỏi, hãy trả lời chính xác nguyên văn câu sau: "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu được cung cấp." và không viết thêm bất kỳ lập luận hoặc phỏng đoán nào khác.
3. TRÍCH DẪN NGUỒN (CITATIONS): Trong câu trả lời của bạn, bất cứ khi nào bạn sử dụng thông tin từ một đoạn tham khảo nào, hãy chú thích nguồn tương ứng ngay cuối câu bằng cách ghi [Trang X] (ví dụ: "...được thiết kế theo cấu trúc [Trang 5]") hoặc [Hệ thống] đối với thông tin thống kê tổng quan của hệ thống.
4. Trình bày câu trả lời bằng tiếng Việt khoa học, mạch lạc, sử dụng danh sách liệt kê (bullet points) khi cần thiết để tăng tính thẩm mỹ và dễ đọc.
"""
    # retry tránh lỗi 503
    for _ in range(3):
        try:
            response = client.models.generate_content(
                model=active_model,
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            if "503" in str(e):
                time.sleep(2)
            else:
                return f"❌ Lỗi kết nối AI: {str(e)}"
    return "❌ Hệ thống AI đang bận hoặc quá tải, vui lòng thử lại sau giây lát."