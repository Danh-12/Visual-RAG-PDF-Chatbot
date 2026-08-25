import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import streamlit as st
import requests
import base64
import hashlib
import json
import os
from config import MODEL_NAME, API_URL

# Format a clean model name for display
model_display_name = MODEL_NAME.replace("models/", "").replace("-", " ").title()

# ===== CONFIG PAGE =====
st.set_page_config(
    page_title="Visual RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== PREMIUM DESIGN SYSTEM (CSS) =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek gradient background for headers */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom stats card */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    
    .stat-lbl {
        font-size: 0.9rem;
        color: #8c8c8c;
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Source citation badges */
    .cite-badge {
        display: inline-block;
        background-color: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 5px;
        padding: 2px 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Kiểm tra kết nối FastAPI Backend
backend_online = False
backend_stats = {}
try:
    response = requests.get(f"{API_URL}/api/stats", timeout=2)
    if response.status_code == 200:
        backend_online = True
        backend_stats = response.json()
except Exception:
    backend_online = False

# ===== STATE INITIALIZATION =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "stats" not in st.session_state:
    st.session_state.stats = {"ready": False, "pages": 0, "chunks": 0, "images": 0}

# Đồng bộ stats từ Backend nếu backend online
if backend_online and backend_stats.get("ready"):
    st.session_state.stats = backend_stats
elif not backend_online and os.path.exists("faiss_index"):
    # Tự động tải trạng thái sẵn sàng nếu chạy offline và đã có DB
    st.session_state.stats["ready"] = True
    if os.path.exists("faiss_index/stats.json"):
        try:
            with open("faiss_index/stats.json", "r", encoding="utf-8") as f:
                st.session_state.stats = {**json.load(f), "ready": True}
        except Exception:
            pass

# ===== SIDEBAR / FILE UPLOAD =====
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8207/8207185.png", width=70) # Beautiful AI assistant logo
    st.markdown("<h2 style='margin-top: 0;'>Visual RAG Panel</h2>", unsafe_allow_html=True)
    st.write("Giải pháp phân tích tài liệu PDF đa phương thức nâng cao (Văn bản + Sơ đồ + Bảng biểu) sử dụng Gemini API.")
    
    st.markdown("---")
    
    # Hiển thị trạng thái kết nối
    if backend_online:
        st.success("🟢 Connected to FastAPI Backend (API Mode)")
    else:
        st.info("💡 FastAPI Backend Offline (Chạy chế độ Local Fallback)")
        
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📥 Tải lên tài liệu PDF", type="pdf")
    
    st.markdown("---")
    st.markdown("### ⚙️ Cấu hình Hệ thống")
    
    model_options = {
        "Gemini 3.1 Flash Lite (Khuyên dùng)": "models/gemini-3.1-flash-lite",
        "Gemini 2.5 Flash Lite (Dự phòng)": "models/gemini-2.5-flash-lite",
        "Gemini 1.5 Flash (Bền bỉ)": "models/gemini-flash-latest",
    }
    
    selected_model_label = st.selectbox(
        "🤖 Mô hình AI (LLM)",
        options=list(model_options.keys()),
        index=0
    )
    active_model = model_options[selected_model_label]
    
    analyze_images = st.checkbox(
        "🖼️ Phân tích Sơ đồ/Bảng biểu", 
        value=True
    )
    max_images = st.slider(
        "🖼️ Số hình ảnh tối đa",
        min_value=1,
        max_value=50,
        value=30
    )
        
    st.info(f"🤖 **LLM**: {selected_model_label}\n\n⚡ **Embeddings**: HuggingFace Embeddings\n\n📌 **Vector Store**: FAISS (Persistent)")

# ===== FILE PROCESSING =====
if uploaded_file:
    file_bytes = uploaded_file.read()
    current_hash = hashlib.md5(file_bytes).hexdigest()
    uploaded_file.seek(0)
    
    if st.session_state.file_hash != current_hash:
        st.session_state.chat_history = []
        st.session_state.file_hash = current_hash
        
        if backend_online:
            # Chạy qua API Backend
            with st.status("📦 Đang upload và xử lý tài liệu PDF trên Backend API...", expanded=True) as status:
                try:
                    files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
                    data = {
                        "analyze_images": str(analyze_images).lower(),
                        "max_images": str(max_images),
                        "active_model": active_model
                    }
                    
                    status.update(label="📄 Đang phân tích PDF và chạy Vision AI...", state="running")
                    response = requests.post(f"{API_URL}/api/upload", files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.session_state.stats = {**res_data["stats"], "ready": True}
                        status.update(label="✅ Hệ thống RAG đã sẵn sàng!", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status.update(label="❌ Lỗi khi tải lên backend!", state="error")
                        st.error(f"Chi tiết lỗi backend: {response.text}")
                except Exception as e:
                    status.update(label="❌ Lỗi kết nối API!", state="error")
                    st.error(f"Không thể kết nối tới FastAPI Backend tại {API_URL}: {e}")
        else:
            # Chạy trực tiếp cục bộ (Local Fallback)
            with st.status("📦 Đang phân tích PDF cục bộ (Local Fallback)...", expanded=True) as status:
                try:
                    import fitz
                    from pdf_utils import extract_documents, extract_images
                    from rag_pipeline import split_documents, generate_image_description, create_vector_db, save_vector_db
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from langchain_core.documents import Document
                    
                    status.update(label="📄 Đang trích xuất văn bản từng trang...", state="running")
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    text_docs = extract_documents(doc)
                    text_chunks = split_documents(text_docs)
                    
                    image_docs = []
                    images = []
                    
                    if analyze_images:
                        status.update(label="🖼️ Đang quét sơ đồ/bảng biểu...", state="running")
                        images = extract_images(doc)
                        
                        if images:
                            num_imgs = len(images)
                            img_progress = st.progress(0, text="Đang phân tích song song hình ảnh bằng Vision AI...")
                            
                            def process_single_image(idx, img):
                                should_describe = idx < max_images
                                desc = ""
                                if should_describe:
                                    try:
                                        desc = generate_image_description(img["image_bytes"], img["ext"], model_name=active_model)
                                        desc = desc.strip()
                                    except Exception:
                                        desc = ""
                                
                                if not desc:
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
                                
                            with ThreadPoolExecutor(max_workers=3) as executor:
                                future_to_img = {executor.submit(process_single_image, idx, img): img for idx, img in enumerate(images)}
                                completed = 0
                                for future in as_completed(future_to_img):
                                    try:
                                        res = future.result()
                                        if res:
                                            image_docs.append(res)
                                    except Exception:
                                        pass
                                    completed += 1
                                    img_progress.progress(completed / num_imgs, text=f"Đang xử lý hình ảnh ({completed}/{num_imgs})...")
                            img_progress.empty()
                            
                    status.update(label="⚡ Đang xây dựng Vector DB FAISS...", state="running")
                    all_chunks = text_chunks + image_docs
                    db = create_vector_db(all_chunks)
                    
                    # Lưu trữ cục bộ xuống đĩa
                    if os.path.exists("faiss_index"):
                        import shutil
                        shutil.rmtree("faiss_index")
                    os.makedirs("faiss_index", exist_ok=True)
                    save_vector_db(db, "faiss_index")
                    
                    # Lưu file thống kê stats.json cục bộ
                    stats_data = {
                        "pages": len(doc),
                        "chunks": len(text_chunks),
                        "images": len(images)
                    }
                    with open("faiss_index/stats.json", "w", encoding="utf-8") as f:
                        json.dump(stats_data, f)
                        
                    st.session_state.stats = {**stats_data, "ready": True}
                    status.update(label="✅ Hệ thống RAG cục bộ đã sẵn sàng!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Lỗi xử lý tài liệu cục bộ!", state="error")
                    st.error(f"Chi tiết lỗi: {e}")

# ===== APP LAYOUT & INTERACTIVE DASHBOARD =====
st.markdown("<h1 class='main-title'>📄 Visual RAG PDF Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Hỏi đáp thông minh vượt giới hạn văn bản - Hỗ trợ phân tích cả hình ảnh, sơ đồ và bảng biểu phức tạp</p>", unsafe_allow_html=True)

if not st.session_state.stats.get("ready"):
    st.info("👈 Vui lòng tải lên một tệp PDF ở thanh bên trái để bắt đầu trò chuyện!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-val'>⚡ Tùy chọn</div>
            <div class='stat-lbl'>Hỗ trợ API Mode & Local Mode</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-val'>🖼️ Multimodal</div>
            <div class='stat-lbl'>Đọc hiểu sơ đồ & bảng biểu tự động</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-val'>🛡️ Persistent</div>
            <div class='stat-lbl'>Lưu trữ Vector DB xuống ổ đĩa</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Hiển thị Dashboard tài liệu
    stats = st.session_state.stats
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats.get('pages', 0)}</div><div class='stat-lbl'>Tổng số trang</div></div>", unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats.get('chunks', 0)}</div><div class='stat-lbl'>Phân đoạn văn bản</div></div>", unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats.get('images', 0)}</div><div class='stat-lbl'>Sơ đồ & Bảng biểu đã xử lý</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Giao diện hội thoại (Chat History)
    for msg in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(msg["query"])
            
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["answer"])
            
            # Hiển thị nguồn trích dẫn & hình ảnh nếu có
            if msg["docs"]:
                with st.expander("📚 Nguồn trích dẫn liên quan"):
                    pages = sorted(list(set([doc["metadata"]["page"] for doc in msg["docs"] if doc["metadata"].get("page") and doc["metadata"].get("page"] > 0])))
                    page_badges = "".join([f"<span class='cite-badge'>Trang {p}</span>" for p in pages])
                    st.markdown(f"**Nguồn từ:** {page_badges}", unsafe_allow_html=True)
                    st.write("---")
                    
                    for i, doc in enumerate(msg["docs"]):
                        page_num = doc["metadata"].get("page", "không rõ")
                        source_type = doc["metadata"].get("source", "text")
                        
                        if source_type == "image":
                            st.markdown(f"**🖼️ Sơ đồ/Bảng biểu (Trang {page_num}):**")
                            st.write(doc["page_content"])
                            if doc["metadata"].get("image_base64"):
                                img_bytes = base64.b64decode(doc["metadata"]["image_base64"])
                                st.image(
                                    img_bytes, 
                                    caption=f"Sơ đồ / Bảng biểu trích xuất từ Trang {page_num}",
                                    use_container_width=True
                                )
                        else:
                            st.markdown(f"**📄 Văn bản tham khảo {i+1} (Trang {page_num}):**")
                            st.write(doc["page_content"])
                        st.write("---")
                        
    # Nhập câu hỏi mới
    query = st.chat_input("💬 Hỏi tôi bất cứ điều gì về nội dung hoặc sơ đồ trong tài liệu này...")
    
    if query:
        with st.chat_message("user", avatar="👤"):
            st.write(query)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Đang phân tích tài liệu và suy luận..."):
                if backend_online:
                    # Giao tiếp API Backend
                    try:
                        clean_history = [{"query": turn["query"], "answer": turn["answer"]} for turn in st.session_state.chat_history[-3:]]
                        payload = {
                            "query": query,
                            "model_name": active_model,
                            "chat_history": clean_history
                        }
                        response = requests.post(f"{API_URL}/api/query", json=payload)
                        if response.status_code == 200:
                            res_data = response.json()
                            answer = res_data["answer"]
                            docs = res_data["docs"]
                        else:
                            answer = f"❌ **Lỗi API backend:** {response.text}"
                            docs = []
                    except Exception as e:
                        answer = f"❌ **Lỗi kết nối tới API server:** {e}"
                        docs = []
                else:
                    # Truy vấn cục bộ (Local Fallback)
                    try:
                        from rag_pipeline import load_vector_db, retrieve_docs, generate_answer
                        db = load_vector_db("faiss_index")
                        retrieved_docs = retrieve_docs(query, db)
                        answer = generate_answer(query, retrieved_docs, model_name=active_model, chat_history=st.session_state.chat_history)
                        
                        # Serialize để thống nhất cấu trúc dữ liệu hiển thị
                        docs = []
                        for doc in retrieved_docs:
                            image_base64 = None
                            if "image_bytes" in doc.metadata and doc.metadata["image_bytes"]:
                                image_base64 = base64.b64encode(doc.metadata["image_bytes"]).decode("utf-8")
                            docs.append({
                                "page_content": doc.page_content,
                                "metadata": {
                                    "source": doc.metadata.get("source", "text"),
                                    "page": doc.metadata.get("page", 0),
                                    "image_base64": image_base64,
                                    "ext": doc.metadata.get("ext")
                                }
                            })
                    except Exception as e:
                        answer = f"❌ **Lỗi truy vấn cục bộ:** {e}"
                        docs = []
                
            st.markdown(answer)
            
            if docs:
                with st.expander("📚 Nguồn trích dẫn liên quan"):
                    pages = sorted(list(set([doc["metadata"]["page"] for doc in docs if doc["metadata"].get("page"] and doc["metadata"].get("page") > 0])))
                    page_badges = "".join([f"<span class='cite-badge'>Trang {p}</span>" for p in pages])
                    st.markdown(f"**Nguồn từ:** {page_badges}", unsafe_allow_html=True)
                    st.write("---")
                    
                    for i, doc in enumerate(docs):
                        page_num = doc["metadata"].get("page", "không rõ")
                        source_type = doc["metadata"].get("source", "text")
                        
                        if source_type == "system":
                            st.markdown(f"**⚡ Thống kê Tổng quan (Hệ thống tự động):**")
                            st.write(doc["page_content"])
                        elif source_type == "image":
                            st.markdown(f"**🖼️ Sơ đồ/Bảng biểu (Trang {page_num}):**")
                            st.write(doc["page_content"])
                            if doc["metadata"].get("image_base64"):
                                img_bytes = base64.b64decode(doc["metadata"]["image_base64"])
                                st.image(
                                    img_bytes, 
                                    caption=f"Sơ đồ / Bảng biểu trích xuất từ Trang {page_num}",
                                    use_container_width=True
                                )
                        else:
                            st.markdown(f"**📄 Văn bản tham khảo {i+1} (Trang {page_num}):**")
                            st.write(doc["page_content"])
                        st.write("---")
                        
        # Lưu lịch sử chat
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "docs": docs
        })
        st.rerun()