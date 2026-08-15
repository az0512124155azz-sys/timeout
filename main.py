"""
Backend ראשי - FastAPI
מריץ 5 endpoints נפרדים, אחד לכל שירות שביקשת:

  POST /api/lipsync         - שירות 2: תיקון שפתיים בלבד (סרטון קיים + שמע קיים)
  POST /api/videogen        - שירות 3: יצירת סרטון מטקסט/תמונה בלבד
  POST /api/tts             - שירות 1: יצירת קול/שיבוט קול בלבד
  POST /api/pipeline/full   - שירות 4: הכל ביחד (טקסט/תמונה -> סרטון -> קול -> סנכרון)
  POST /api/pipeline/voice-lipsync - שירות 5: סרטון קיים + טקסט -> קול חדש + סנכרון
                                      (זה מה שביקשת בהתחלה - הסרטון AI + טקסט + הגדרת קול)

⚠️ הרצה: זה חייב לרוץ על מכונה עם GPU (מקומית או שרת ענן שכור).
לא ניתן להריץ את ה-backend הזה על Vercel - ראו README.md.

הרצה מקומית:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services import tts_service, lipsync_service, videogen_service
from services import pipeline_full, pipeline_voice_lipsync

app = FastAPI(title="AI Voice & Video Studio")

# מאפשר לפרונטאנד (גם אם רץ על דומיין אחר, למשל Vercel) לקרוא ל-backend הזה
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # בפרודקשן כדאי להגביל לדומיין הספציפי של הפרונטאנד שלך
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", "./outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file: UploadFile, suffix: str) -> str:
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/voice-presets")
def voice_presets():
    """מחזיר את רשימת הקולות המוכנים הזמינים (למלא ב-voice_presets/)."""
    return {"presets": tts_service.list_voice_presets()}


# ---------- שירות 1: TTS / שיבוט קול בלבד ----------
@app.post("/api/tts")
async def api_tts(
    text: str = Form(...),
    language: str = Form("he"),
    voice_preset: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    try:
        ref_path = _save_upload(reference_audio, "ref.wav") if reference_audio else None
        audio_path = tts_service.synthesize_speech(
            text=text,
            language=language,
            speaker_wav_path=ref_path,
            voice_preset=voice_preset,
            voice_description=voice_description,
        )
        return FileResponse(audio_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- שירות 2: תיקון שפתיים בלבד ----------
@app.post("/api/lipsync")
async def api_lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    try:
        video_path = _save_upload(video, "video.mp4")
        audio_path = _save_upload(audio, "audio.wav")
        result_path = lipsync_service.apply_lipsync(video_path, audio_path)
        return FileResponse(result_path, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- שירות 3: יצירת סרטון מטקסט/תמונה בלבד ----------
@app.post("/api/videogen")
async def api_videogen(
    prompt: str = Form(...),
    generate_type: str = Form("t2v"),
    source_image: Optional[UploadFile] = File(None),
):
    try:
        if generate_type == "i2v":
            if not source_image:
                raise ValueError("generate_type='i2v' דורש source_image")
            image_path = _save_upload(source_image, "src.png")
            video_path = videogen_service.generate_video_from_image(image_path, prompt)
        else:
            video_path = videogen_service.generate_video_from_text(prompt)
        return FileResponse(video_path, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- שירות 4: הפייפליין המלא (הכל ביחד) ----------
@app.post("/api/pipeline/full")
async def api_pipeline_full(
    spoken_text: str = Form(...),
    video_prompt: str = Form(...),
    generate_type: str = Form("t2v"),
    language: str = Form("he"),
    voice_preset: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None),
    source_image: Optional[UploadFile] = File(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    try:
        image_path = _save_upload(source_image, "src.png") if source_image else None
        ref_path = _save_upload(reference_audio, "ref.wav") if reference_audio else None
        result = pipeline_full.run(
            spoken_text=spoken_text,
            video_prompt=video_prompt,
            generate_type=generate_type,
            source_image_path=image_path,
            language=language,
            speaker_wav_path=ref_path,
            voice_preset=voice_preset,
            voice_description=voice_description,
        )
        return FileResponse(result["final_video_path"], media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- שירות 5: סרטון קיים + טקסט -> קול חדש + סנכרון (הבקשה המקורית שלך) ----------
@app.post("/api/pipeline/voice-lipsync")
async def api_pipeline_voice_lipsync(
    video: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("he"),
    voice_preset: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    try:
        video_path = _save_upload(video, "video.mp4")
        ref_path = _save_upload(reference_audio, "ref.wav") if reference_audio else None
        result = pipeline_voice_lipsync.run(
            video_path=video_path,
            text=text,
            language=language,
            speaker_wav_path=ref_path,
            voice_preset=voice_preset,
            voice_description=voice_description,
        )
        return FileResponse(result["final_video_path"], media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# משרת את הפרונטאנד הסטטי (index.html) אם רצים הכל מאותה מכונה
if Path("../frontend").exists():
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
