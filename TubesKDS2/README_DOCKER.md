# Docker — Cell Cycle Intelligence System

Cara paling gampang untuk menjalankan web app tanpa setup Python lokal.

## Quick Start

### 1. Build image

Jalankan dari folder `TubesKDS2/`:

```bash
docker build -t tubeskds2:latest .
```

> **Catatan:** Build akan download PyTorch + dependencies (~2 GB). Hanya perlu sekali.

### 2. Run

```bash
docker run --rm -p 5050:5050 tubeskds2:latest
```

Buka browser: **http://localhost:5050**

Fitur yang langsung tersedia:
- ✅ **Live Inference** — upload gambar sel → klasifikasi + Grad-CAM
- ✅ **ODE Dynamics** — simulasi Cyclin-CDK
- ✅ **HMM Correction** — demo Viterbi decoding
- ✅ **Checkpoint Detector** — anomaly detection

### 3. Run (full dashboard + dataset)

Kalau mau dashboard CNN evaluation (butuh dataset test set):

```bash
docker run --rm -p 5050:5050 \
  -e LOAD_DATA=1 \
  -v "$PWD/data:/app/data" \
  tubeskds2:latest
```

### 4. Health check

```bash
curl -s http://localhost:5050/api/status | python3 -m json.tool
```

Kalau `model_loaded` = `false`, berarti file `output/models/best_model.pth` tidak ada.
