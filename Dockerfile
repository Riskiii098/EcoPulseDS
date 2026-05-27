# Gunakan base image Python yang ringan dan stabil
FROM python:3.12-slim

# Menghindari interaksi selama instalasi paket sistem
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory di dalam container
WORKDIR /app

# Install dependensi sistem dasar untuk kompilasi library ML dan dependensinya
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk caching layer Docker yang optimal
COPY requirements.txt .

# Install seluruh Python dependencies (termasuk scikit-learn, mlflow, xgboost, dll)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh struktur direktori proyek ke dalam container
COPY . .

# Konfigurasi MLflow URI environment variables
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# Ekspos port yang dibutuhkan:
# 5000: MLflow UI
# 8888: Jupyter Notebook
# 7860: Port wajib untuk Hugging Face Spaces
EXPOSE 7860

# Perintah default saat container dijalankan
CMD ["python", "app.py"]
