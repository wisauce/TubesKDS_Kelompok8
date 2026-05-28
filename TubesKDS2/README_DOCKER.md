# Docker Run (Cell Cycle Intelligence System)

Ini cara paling gampang supaya orang lain bisa coba tanpa setup Python lokal.

## 1) Build image

Jalankan dari folder `TubesKDS2/`:

```zsh
docker build -t tubeskds2:latest .
```

Catatan: build akan download `torch/torchvision` (cukup besar).

## 2) Run (inference-only)

Butuh model di `output/models/best_model.pth`.

Opsi paling gampang untuk platform seperti Railway:
- set env `MODEL_URL` (direct link) supaya server download weights saat startup, dan
- set `LOAD_DATA=0` supaya tidak perlu dataset test set.

```zsh
docker run --rm -p 5050:5050 \
  -e PORT=5050 \
  -e LOAD_DATA=0 \
  -v "$PWD/output:/app/output" \
  tubeskds2:latest
```

Atau (tanpa mount output) kalau kamu menyediakan URL weights:

```zsh
docker run --rm -p 5050:5050 \
  -e PORT=5050 \
  -e LOAD_DATA=0 \
  -e MODEL_URL="https://<direct-link>/best_model.pth" \
  tubeskds2:latest
```

Buka:
- http://127.0.0.1:5050

## 3) Run (full dashboard)

Selain model, butuh dataset processed di `data/synthetic/`.

```zsh
docker run --rm -p 5050:5050 \
  -e PORT=5050 \
  -v "$PWD/output:/app/output" \
  -v "$PWD/data:/app/data" \
  tubeskds2:latest
```

## 4) Health check

```zsh
curl -s http://127.0.0.1:5050/api/status
```

Kalau `model_loaded` / `data_loaded` masih `false`, berarti volume mount belum benar atau foldernya belum ada di host.
