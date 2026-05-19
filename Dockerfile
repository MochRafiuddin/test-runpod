# Menggunakan base image Python yang ringan
FROM python:3.10-slim

# Menentukan *working directory* di dalam container
WORKDIR /app

# Menginstall library runpod
RUN pip install --no-cache-dir runpod

# Menyalin file aplikasi ke dalam container
COPY app.py .

# Menjalankan aplikasi dengan flag -u agar log langsung muncul di RunPod (unbuffered)
CMD [ "python", "-u", "app.py" ]