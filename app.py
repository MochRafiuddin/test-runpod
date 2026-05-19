import runpod
import requests
from ultralytics import YOLO
import os
import datetime

# Me-load model custom milikmu yang sudah tertanam di image
model = YOLO("model_chompchomp_new.pt")

def handler(job):
    # Tentukan nama file secara konsisten dengan ekstensi .jpg
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_image_path = f"temp_input_{timestamp}.jpg"
    
    try:
        # Mengambil input data dari request pengguna
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        
        if not image_url:
            return {"status": "error", "message": "Input 'image_url' tidak ditemukan."}
        
        # 1. Download gambar sementara dari URL dengan menambahkan Headers Browser umum
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, stream=True, timeout=15)
        
        if response.status_code == 200:
            with open(temp_image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            return {"status": "error", "message": f"Gagal mendownload gambar dari URL. Status code: {response.status_code}"}
        
        # 2. Jalankan deteksi objek dengan YOLO
        results = model(temp_image_path)
        
        # 3. Parsing hasil deteksi
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Mengambil koordinat bounding box, confidence score, dan class ID
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "object": class_name,
                    "confidence": confidence,
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
        
        # Hapus file temporary setelah sukses agar container tetap bersih
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        return {
            "requests": job_input,
            "status": "success",
            "total_detected": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        # Jika terjadi eror di tengah jalan (misal: gagal parsing), pastikan file temporary tetap dihapus
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"status": "error", "message": str(e)}

# Jalankan RunPod serverless worker
runpod.serverless.start({"handler": handler})