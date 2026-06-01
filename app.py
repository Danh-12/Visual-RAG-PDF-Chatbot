import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import streamlit as st
import fitz  # PyMuPDF
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from pdf_utils import extract_documents, extract_images
from rag_pipeline import split_documents, generate_image_description, create_vector_db, retrieve_docs, generate_answer
from config import MODEL_NAME
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
# ===== SIDEBAR / FILE UPLOAD =====
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8207/8207185.png", width=70) # Beautiful AI assistant logo
    st.markdown("<h2 style='margin-top: 0;'>Visual RAG Panel</h2>", unsafe_allow_html=True)
    st.write("Giải pháp phân tích tài liệu PDF đa phương thức nâng cao (Văn bản + Sơ đồ + Bảng biểu) sử dụng các mô hình Gemini thế hệ mới.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📥 Tải lên tài liệu PDF", type="pdf")
    
    st.markdown("---")
    st.markdown("### ⚙️ Cấu hình Hệ thống")
    
    # Lựa chọn mô hình thông minh và hiển thị đúng hạn ngạch (quota) tránh lỗi 429
    model_options = {
        "Gemini 3.1 Flash Lite (Khuyên dùng - Đang HOẠT ĐỘNG cực tốt)": "models/gemini-3.1-flash-lite",
        "Gemini 2.5 Flash Lite (Dự phòng 1 - Đang HOẠT ĐỘNG ổn định)": "models/gemini-2.5-flash-lite",
        "Gemini 2.5 Flash (Tạm thời hết lượt hôm nay - Lỗi 429)": "models/gemini-2.5-flash",
        "Gemini 2.0 Flash (Tạm thời hết lượt hôm nay - Lỗi 429)": "models/gemini-2.0-flash",
        "Gemini 1.5 Flash (Dự phòng 2 - Bền bỉ)": "models/gemini-flash-latest",
        "Gemini 3.5 Flash (Tạm thời hết lượt hôm nay - Lỗi 429)": "models/gemini-3.5-flash",
        "Gemini 2.5 Pro (Tạm thời hết lượt hôm nay - Lỗi 429)": "models/gemini-2.5-pro",
    }
    
    selected_model_label = st.selectbox(
        "🤖 Mô hình AI (LLM)",
        options=list(model_options.keys()),
        index=0,
        help="Hãy chọn Gemini 3.1 Flash Lite hoặc Gemini 2.5 Flash Lite vì các mô hình này hiện đang hoạt động và còn nguyên hạn ngạch lượt dùng!"
    )
    active_model = model_options[selected_model_label]
    
    analyze_images = st.checkbox(
        "🖼️ Phân tích Sơ đồ/Bảng biểu", 
        value=True, 
        help="Bật tính năng này để Gemini tự động đọc và phân tích các sơ đồ, hình vẽ và bảng biểu trong tài liệu. Tắt đi nếu muốn quét cực nhanh chỉ lấy văn bản."
    )
    max_images = 30
    if analyze_images:
        max_images = st.slider(
            "🖼️ Số hình ảnh tối đa để phân tích",
            min_value=1,
            max_value=50,
            value=30,
            help="Giới hạn số hình ảnh phân tích để tránh vượt quá giới hạn Rate Limit (15 RPM) của tài khoản Free Tier."
        )
        
    st.info(f"🤖 **LLM**: {selected_model_label}\n\n⚡ **Embeddings**: HuggingFace Embeddings (all-MiniLM-L6-v2 - Cục bộ/Offline)\n\n📌 **Vector Store**: FAISS")
    
    st.markdown("---")
    st.markdown("### 💡 Mẹo đặt câu hỏi RAG")
    st.info(
        "**Tài liệu Kỷ yếu rất lớn (gần 300 trang) chứa nhiều đề tài khác nhau.**\n\n"
        "Để hệ thống RAG định vị và trích dẫn chính xác trang sách bạn cần, **tránh đặt câu hỏi quá chung chung**. Hãy đặt câu hỏi cụ thể kèm từ khóa hoặc tên tác giả:\n\n"
        "* ❌ *Chung chung*: 'Đề tài báo cáo cuối kỳ này tên là gì?'\n"
        "*  *Cụ thể*: 'Đề tài của sinh viên Nguyễn Thành Danh tên là gì?' hoặc 'Tóm tắt nội dung hệ thống Online Appointment Booking System ở trang 1/trang 4'."
    )
# ===== STATE INITIALIZATION =====
if "db" not in st.session_state:
    st.session_state.db = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stats" not in st.session_state:
    st.session_state.stats = {}
# ===== FILE PROCESSING =====
if uploaded_file:
    # Đọc dữ liệu file
    file_bytes = uploaded_file.read()
    
    # Tính hash của file để phát hiện thay đổi
    current_hash = hashlib.md5(file_bytes).hexdigest()
    uploaded_file.seek(0) # Reset stream đầu vào
    
    # Nếu tải file mới lên, giải phóng tài nguyên cũ và xây dựng lại DB
    if st.session_state.file_hash != current_hash:
        st.session_state.chat_history = []
        st.session_state.db = None
        st.session_state.file_hash = current_hash
        
        # Bắt đầu xử lý file PDF
        with st.status("📦 Đang phân tích và xử lý PDF đa phương thức...", expanded=True) as status:
            # 1. Đọc PDF bằng PyMuPDF
            status.update(label="📄 Đang trích xuất văn bản từng trang...", state="running")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Trích xuất văn bản
            text_docs = extract_documents(doc)
            text_chunks = split_documents(text_docs)
            st.write(f"✓ Đã trích xuất và phân tách thành **{len(text_chunks)}** phân đoạn văn bản.")
            
            # 2. Trích xuất hình ảnh
            image_docs = []
            images = []
            
            if analyze_images:
                status.update(label="🖼️ Đang quét tìm kiếm hình ảnh, sơ đồ, bảng biểu...", state="running")
                images = extract_images(doc)
                
                if images:
                    st.write(f"✓ Bộ lọc thông minh đã tự động lọc bỏ các logo và biểu tượng nhỏ trang trí.")
                    st.write(f"🎯 Phát hiện và trích xuất thành công **{len(images)}** sơ đồ/hình ảnh kỹ thuật chất lượng cao.")
                    
                    original_num_imgs = len(images)
                    if original_num_imgs > max_images:
                        st.warning(f"⚠️ Phát hiện **{original_num_imgs}** sơ đồ/hình ảnh kỹ thuật. Chỉ phân tích tối đa **{max_images}** hình bằng Vision AI theo cấu hình để tiết kiệm quota, các hình còn lại vẫn được nạp đầy đủ ở dạng ảnh gốc thô.")
                        
                    num_imgs = len(images)
                    
                    # Thanh tiến trình phân tích hình ảnh
                    img_progress = st.progress(0, text="Đang phân tích song song hình ảnh bằng Vision AI...")

                    # Hàm con xử lý từng ảnh trong luồng phụ
                    def process_single_image(idx, img):
                        should_describe = idx < max_images
                        desc = ""
                        
                        if should_describe:
                            try:
                                desc = generate_image_description(img["image_bytes"], img["ext"], model_name=active_model)
                                desc = desc.strip()
                            except Exception as e:
                                print(f"Lỗi gọi API phân tích hình ảnh trang {img['page']}: {e}")
                                desc = ""
                        
                        # Nếu không được phân tích AI hoặc phân tích thất bại (lỗi rate limit)
                        # Tạo bản mô tả dự phòng (fallback) để giữ lại ảnh trong Vector Store và hiển thị trên UI
                        if not desc:
                            if should_describe:
                                desc = (f"Sơ đồ thực tế tại trang {img['page']}. "
                                        f"(Ghi chú: Bản mô tả chi tiết bằng AI tạm thời chưa tải được do giới hạn quota API hoặc lỗi kết nối, "
                                        f"tuy nhiên sơ đồ gốc vẫn được lưu trữ đầy đủ và sẵn sàng hiển thị khi bạn đặt câu hỏi liên quan).")
                            else:
                                desc = (f"Sơ đồ thực tế tại trang {img['page']}. "
                                        f"(Ghi chú: Sơ đồ gốc được lưu trữ và sẵn sàng hiển thị trực quan cho bạn).")
                                        
                        return Document(
                            page_content=f"[Sơ đồ/Bảng biểu ở trang {img['page']}]:\n{desc}",
                            metadata={
                                "source": "image",
                                "page": img["page"],
                                "image_bytes": img["image_bytes"],
                                "ext": img["ext"]
                            }
                        )

                    # Sử dụng ThreadPoolExecutor để chạy song song cực nhanh và ổn định
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_img = {executor.submit(process_single_image, idx, img): img for idx, img in enumerate(images)}
                        
                        completed = 0
                        for future in as_completed(future_to_img):
                            img = future_to_img[future]
                            try:
                                res = future.result()
                                if res:
                                    image_docs.append(res)
                            except Exception as e:
                                print(f"Lỗi xử lý luồng hình ảnh tại trang {img['page']}: {e}")
                            
                            completed += 1
                            img_progress.progress(
                                completed / num_imgs,
                                text=f"Đang xử lý luồng hình ảnh ({completed}/{num_imgs})..."
                            )

                    img_progress.empty()  # Xóa thanh tiến trình
                    st.write(f"✓ Đã hoàn thành xử lý luồng hình ảnh: **{len(image_docs)}** sơ đồ kỹ thuật đã được nạp thành công.")
                else:
                    st.write("⏭️ Chưa có sơ đồ/bảng biểu nào được phát hiện trong PDF.")
            else:
                st.write("⏭️ Đã bỏ qua phân tích hình ảnh (chỉ quét văn bản).")

            # 3. Kết hợp và Vectorize
            status.update(label="⚡ Đang số hóa tài liệu bằng HuggingFace Offline Embeddings...", state="running")
            all_chunks = text_chunks + image_docs
            
            # Tạo DB
            db = create_vector_db(all_chunks)
            
            # Lưu trữ vào session_state
            st.session_state.db = db
            st.session_state.stats = {
                "pages": len(doc),
                "chunks": len(text_chunks),
                "images": len(images)
            }
            
            status.update(label="✅ Hệ thống RAG đa phương thức đã sẵn sàng!", state="complete", expanded=False)
            st.rerun()
# ===== APP LAYOUT & INTERACTIVE DASHBOARD =====
st.markdown("<h1 class='main-title'>📄 Visual RAG PDF Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Hỏi đáp thông minh vượt giới hạn văn bản - Hỗ trợ phân tích cả hình ảnh, sơ đồ và bảng biểu phức tạp</p>", unsafe_allow_html=True)
if st.session_state.db is None:
    # Chế độ chờ tải file
    st.info("👈 Vui lòng tải lên một tệp PDF ở thanh bên trái để bắt đầu trò chuyện!")
    
    # Showcase một số tính năng đẹp mắt bằng CSS
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-val'>⚡ <span style='font-size:1.2rem;'>Trực tuyến</span></div>
            <div class='stat-lbl'>Gemini Cloud Embeddings siêu tốc</div>
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
            <div class='stat-val'>🛡️ An toàn</div>
            <div class='stat-lbl'>Chống Hallucination nghiêm ngặt</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Hiển thị Dashboard tài liệu
    stats = st.session_state.stats
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats['pages']}</div><div class='stat-lbl'>Tổng số trang</div></div>", unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats['chunks']}</div><div class='stat-lbl'>Phân đoạn văn bản</div></div>", unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{stats['images']}</div><div class='stat-lbl'>Sơ đồ & Bảng biểu đã xử lý</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Giao diện hội thoại (Chat History)
    for msg in st.session_state.chat_history:
        # Câu hỏi của user
        with st.chat_message("user", avatar="👤"):
            st.write(msg["query"])
            
        # Câu trả lời của AI
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["answer"])
            
            # Hiển thị nguồn trích dẫn & hình ảnh nếu có
            if msg["docs"]:
                with st.expander("📚 Nguồn trích dẫn liên quan"):
                    # Lấy danh sách trang trích dẫn (Unique, bỏ qua trang 0 của hệ thống)
                    pages = sorted(list(set([doc.metadata.get("page") for doc in msg["docs"] if doc.metadata.get("page") and doc.metadata.get("page") > 0])))
                    page_badges = "".join([f"<span class='cite-badge'>Trang {p}</span>" for p in pages])
                    st.markdown(f"**Nguồn từ:** {page_badges}", unsafe_allow_html=True)
                    st.write("---")
                    
                    # Hiển thị nội dung chi tiết của nguồn
                    for i, doc in enumerate(msg["docs"]):
                        page_num = doc.metadata.get("page", "không rõ")
                        source_type = doc.metadata.get("source", "text")
                        
                        if source_type == "image":
                            st.markdown(f"**🖼️ Sơ đồ/Bảng biểu (Trang {page_num}):**")
                            # Hiển thị mô tả văn bản
                            st.write(doc.page_content)
                            # Hiển thị trực tiếp hình ảnh gốc trích xuất từ PDF!
                            st.image(
                                doc.metadata["image_bytes"], 
                                caption=f"Sơ đồ / Bảng biểu trích xuất từ Trang {page_num}",
                                use_container_width=True
                            )
                        else:
                            st.markdown(f"**📄 Văn bản tham khảo {i+1} (Trang {page_num}):**")
                            st.write(doc.page_content)
                        st.write("---")
    # Nhập câu hỏi mới
    query = st.chat_input("💬 Hỏi tôi bất cứ điều gì về nội dung hoặc sơ đồ trong tài liệu này...")
    
    if query:
        # 1. Hiển thị câu hỏi của user trên UI ngay lập tức
        with st.chat_message("user", avatar="👤"):
            st.write(query)
            
        # 2. Xử lý câu trả lời
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Đang phân tích tài liệu và suy luận..."):
                try:
                    # Gọi RAG pipeline
                    docs = retrieve_docs(query, st.session_state.db)
                    if not docs:
                        st.warning("⚠️ Không tìm thấy đoạn văn nào trong Vector Store tương khớp với câu hỏi của bạn.")
                    answer = generate_answer(query, docs, model_name=active_model, chat_history=st.session_state.chat_history)
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    docs = []
                    answer = f"❌ **Lỗi hệ thống RAG Pipeline:** {str(e)}\n\n```python\n{tb_str}\n```"
                
            # Hiển thị câu trả lời mới sinh ra
            st.markdown(answer)
            
            # Hiển thị nguồn trích dẫn
            with st.expander("📚 Nguồn trích dẫn liên quan"):
                pages = sorted(list(set([doc.metadata.get("page") for doc in docs if doc.metadata.get("page") and doc.metadata.get("page") > 0])))
                page_badges = "".join([f"<span class='cite-badge'>Trang {p}</span>" for p in pages])
                st.markdown(f"**Nguồn từ:** {page_badges}", unsafe_allow_html=True)
                st.write("---")
                
                for i, doc in enumerate(docs):
                    page_num = doc.metadata.get("page", "không rõ")
                    source_type = doc.metadata.get("source", "text")
                    
                    if source_type == "system":
                        st.markdown(f"**⚡ Thống kê Tổng quan (Hệ thống tự động):**")
                        st.write(doc.page_content)
                    elif source_type == "image":
                        st.markdown(f"**🖼️ Sơ đồ/Bảng biểu (Trang {page_num}):**")
                        st.write(doc.page_content)
                        st.image(
                            doc.metadata["image_bytes"], 
                            caption=f"Sơ đồ / Bảng biểu trích xuất từ Trang {page_num}",
                            use_container_width=True
                        )
                    else:
                        st.markdown(f"**📄 Văn bản tham khảo {i+1} (Trang {page_num}):**")
                        st.write(doc.page_content)
                    st.write("---")
                    
        # 3. Lưu lịch sử chat
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "docs": docs
        })
        st.rerun()