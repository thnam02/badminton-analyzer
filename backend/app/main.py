# Pillow shim before any Detectron2 / DensePose imports (Image.LINEAR removed in Pillow 10+).
from app.cv.densepose.compat import apply_pillow_shims

apply_pillow_shims()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import analyze, health

app = FastAPI(title="Badminton Pose Analyzer", version="0.1.0")

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)

app.mount(
    "/outputs",
    StaticFiles(directory=str(settings.output_dir)),
    name="outputs",
)
