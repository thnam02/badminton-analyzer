# Badminton Pose Analyzer

Minimal scaffold: upload a badminton video → RTMPose (MMPose) → skeleton overlay → browser-playable MP4.

## Structure

```text
badminton-analytics/
├── frontend/          # Next.js (TypeScript + Tailwind)
├── backend/           # FastAPI + OpenCV + MMPose
├── uploads/           # temporary uploads
├── outputs/           # processed videos
├── .env.example
└── README.md
```

## 1. Install frontend dependencies

```bash
cd frontend
npm install
```

## 2. Install Python dependencies

On macOS/CPU, use the bootstrap script (builds `mmcv` with ops when possible, installs MMPose, and applies small compatibility patches):

```bash
cd backend
./scripts/bootstrap_mmpose.sh
source .venv/bin/activate
```

Manual alternative (same pins the script uses):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U 'pip' 'setuptools<81'
pip install -r requirements.txt
MMCV_WITH_OPS=1 pip install 'mmcv==2.1.0' --no-build-isolation
pip install 'mmdet==3.3.0'
pip install 'mmpose==1.3.2' --no-deps
pip install json-tricks munkres
pip install -e ./vendor/xtcocotools_shim
pip install 'chumpy @ git+https://github.com/mattloper/chumpy.git' --no-build-isolation
python scripts/patch_mmpose_for_mmcv_lite.py
```

Also install [FFmpeg](https://ffmpeg.org/) on your PATH (used to produce H.264 output for browsers).

## 3. Configure MMPose model

```bash
cp .env.example .env
```

Edit `.env`:

```text
MMPOSE_CONFIG=          # optional path to RTMPose config .py
MMPOSE_CHECKPOINT=      # optional path to checkpoint .pth
DEVICE=cpu              # or cuda:0
POSE_CONFIDENCE_THRESHOLD=0.5
```

If `MMPOSE_CONFIG` / `MMPOSE_CHECKPOINT` are empty, the backend uses the MMPose alias `human` (RTMPose-m; weights download on first run).

## 4. Run FastAPI

From the `backend` directory (with venv active):

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## 5. Run Next.js

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Optional: point the UI at another API host:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 6. Upload a video

1. Choose an `.mp4` or `.mov` file.
2. Confirm the preview.
3. Click **Analyze Video**.
4. Wait for processing (CPU is slow on long clips).
5. Play the returned skeleton video.

API contract:

- `GET /health` → `{ "status": "ok" }`
- `POST /analyze` (multipart field `video`) → `{ "video_url": "/outputs/....mp4", "output_path": "..." }`
- Processed files are served under `/outputs/...`

## Notes

- Processing is **synchronous** for this scaffold (no queue/Celery).
- Out of scope for v1: auth, DB, stroke classification, shuttle/racket tracking, scoring, LLMs.
