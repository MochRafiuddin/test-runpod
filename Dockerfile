# Menggunakan base image Python yang stabil
FROM python:3.10-slim

# Install library sistem yang dibutuhkan oleh OpenCV & PyTorch di Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install paket Python yang dibutuhkan
RUN pip install --no-cache-dir runpod ultralytics requests

# Menyalin file aplikasi ke dalam container
COPY app.py .

# Trigger download model di awal saat proses BUILD container (agar cold start lebih cepat)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD [ "python", "-u", "app.py" ]