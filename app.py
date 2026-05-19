import runpod

def handler(job):
    """
    Handler utama yang akan dipanggil oleh RunPod setiap kali endpoint di-trigger.
    'job' berisi data input yang dikirim oleh pengguna.
    """
    # Mengambil data dari payload 'input'
    job_input = job.get('input', {})
    name = job_input.get('name', 'Developer')
    
    # Logika test sederhana
    response = {
        "status": "success",
        "message": f"Halo {name}! Serverless RunPod kamu berjalan dengan sempurna."
    }
    
    return response

# Memulai worker serverless RunPod
runpod.serverless.start({"handler": handler})