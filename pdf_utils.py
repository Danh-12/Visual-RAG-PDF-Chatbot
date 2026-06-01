import fitz  # PyMuPDF
from langchain_core.documents import Document
def extract_documents(doc):
    """
    Trích xuất văn bản từ fitz.Document và trả về danh sách LangChain Document
    kèm theo số trang (1-indexed) trong metadata.
    """
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": "text", "page": page_num + 1}
            ))
    return documents
def extract_images(doc):
    """
    Trích xuất hình ảnh chất lượng từ fitz.Document (bỏ qua ảnh trùng lặp hoặc quá nhỏ).
    Trả về danh sách các dict chứa dữ liệu ảnh và số trang tương ứng.
    """
    images = []
    seen_xrefs = set()
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                
                # Bỏ qua những ảnh dạng logo, icon trang trí hoặc đường kẻ phụ trợ nhỏ
                # Một sơ đồ, biểu đồ thực tế luôn có ít nhất chiều rộng hoặc chiều cao lớn hơn hoặc bằng 250px
                if (width < 250 and height < 250) or len(image_bytes) < 3000:
                    continue
                
                images.append({
                    "page": page_num + 1,
                    "image_bytes": image_bytes,
                    "ext": image_ext,
                    "xref": xref
                })
            except Exception as e:
                print(f"Image extraction error at page {page_num + 1}, xref {xref}: {e}")
                
    return images