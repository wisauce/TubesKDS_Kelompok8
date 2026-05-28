                # Deploy on Railway (Inference-only + optional static plots)

Target: 1 service (backend Flask/Gunicorn that also serves the frontend).

## What to ship

- Code in `TubesKDS2/`
- Model weights will be downloaded at runtime using `MODEL_URL`.
- Optional: commit static dashboard plots in `api/static/plots/` (already supported by `/assets/...`).

## 1) Upload model weights somewhere

Recommended: GitHub Release asset.

- Create a release in your repo
- Upload `best_model.pth`
- Copy the **direct download** URL

Notes:
- The URL must be directly downloadable by `curl`/browser (no login page).

## 2) Create Railway project

- New Project → Deploy from GitHub Repo
- Root directory: `TubesKDS2` (so Railway builds using `TubesKDS2/Dockerfile`)

## 3) Set environment variables

Required:
- `MODEL_URL` = direct link to `best_model.pth`

Recommended:
- `LOAD_DATA` = `0` (inference-only, no dataset needed)
- `LOAD_ODE_HMM` = `1` (default; set `0` if you want to skip ODE/HMM init)
- `WEB_CONCURRENCY` = `1` (default; raise only if you have enough RAM)

Optional integrity:
- `MODEL_SHA256` = sha256 of the file (prevents serving corrupted/wrong weights)

## 4) Verify

Open the Railway service URL:
- `/` should show the UI
- `/api/status` should return JSON with `model_loaded: true`
- `/assets/plots/manifest.json` should load if you committed static plots

## Troubleshooting

- If it keeps restarting: check logs (most common is wrong `MODEL_URL`).
- If `model_loaded: false`: verify `MODEL_URL` is reachable and points to the raw file.
