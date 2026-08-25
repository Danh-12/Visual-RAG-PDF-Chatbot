# Sử dụng base image Python 3.11 slim
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết để cài đặt dependencies và chạy ứng dụng
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép và cài đặt các dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn dự án vào container
COPY . .

# Cấp quyền thực thi cho start.sh
RUN chmod +x start.sh

# Expose cổng 8000 (FastAPI Backend) và 8501 (Streamlit Frontend)
EXPOSE 8000
EXPOSE 8501

# Thiết lập các biến môi trường mặc định
ENV PYTHONUNBUFFERED=1

# Lệnh khởi chạy chính
CMD ["./start.sh"]
