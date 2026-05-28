# Deploy on Render (Inference-only)

This project serves the frontend from the Flask backend, so you deploy **one** web service.

## Key differences vs Railway (practical)

- Render free tier typically **sleeps** on inactivity → first request can be slow (cold start + model download if not cached).
- Build context matters: this repo’s `Dockerfile` is inside `TubesKDS2/`, so you should deploy with **root directory = `TubesKDS2`**.
- Render sets `PORT` automatically; this app already binds to `$PORT`.

## 1) Host the model file (best_model.pth)

Recommended: GitHub Release asset.

You will need a **direct download** URL to the file.

## 2) Deploy

Option A (recommended): Blueprint
- Push `render.yaml` (already included in repo root)
- In Render: **New → Blueprint** → select this repo

Option B: Manual Docker web service
- New → Web Service → pick your repo
- **Environment**: Docker
- **Root Directory**: `TubesKDS2`

## 3) Set environment variables

Required:
- `MODEL_URL` = direct URL to `best_model.pth`

Recommended:
- `LOAD_DATA=0` (inference-only)
- `WEB_CONCURRENCY=1` (keeps memory low)

Optional integrity:
- `MODEL_SHA256` = sha256 of `best_model.pth`

## 4) Verify

- `https://<render-url>/` loads UI
- `https://<render-url>/api/status` returns JSON and `model_loaded: true`

If `model_loaded` is false:
- check `MODEL_URL` is a raw file URL (not an HTML page)
