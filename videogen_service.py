"""
שירות 3: יצירת סרטון מטקסט/תמונה (Text/Image-to-Video)
מבוסס על zai-org/CogVideo (CogVideoX) - נטען דרך ספריית diffusers של Hugging Face

חשוב לדעת:
- המשקלים של CogVideoX (5B פרמטרים) שוקלים כ-20-30GB, ומורדים אוטומטית מ-Hugging Face
  בפעם הראשונה שמריצים (דורש אינטרנט + מקום דיסק פנוי).
- דורש GPU עם לפחות ~16-24GB VRAM (עם cpu_offload; בלי זה - הרבה יותר).
- זמן יצירה לסרטון קצר (כמה שניות של וידאו) יכול לקחת כמה דקות גם על GPU חזק.
- זה השירות הכי "יקר" מבין הארבעה מבחינת חומרה.
"""

import os
import uuid
from pathlib import Path
from typing import Literal, Optional

OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# מודלים זמינים (נטענים מ-Hugging Face לפי השם - לא צריך קובץ מקומי)
MODEL_T2V = "THUDM/CogVideoX-5b"          # text-to-video
MODEL_I2V = "THUDM/CogVideoX-5b-I2V"      # image-to-video

_pipe_cache: dict = {}  # קאש כדי לא לטעון מחדש כל פעם


def _get_pipeline(generate_type: Literal["t2v", "i2v"]):
    from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
    import torch

    key = generate_type
    if key in _pipe_cache:
        return _pipe_cache[key]

    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(MODEL_T2V, torch_dtype=torch.bfloat16)
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_I2V, torch_dtype=torch.bfloat16)

    # cpu_offload מוריד דרישת VRAM במחיר מהירות - חשוב על GPU "קטן" (למשל 24GB ומטה)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _pipe_cache[key] = pipe
    return pipe


def generate_video_from_text(
    prompt: str,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
) -> str:
    """יוצר סרטון חדש מתיאור טקסטואלי בלבד (text-to-video)."""
    from diffusers.utils import export_to_video

    pipe = _get_pipeline("t2v")
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    video_frames = result.frames[0]

    out_path = OUTPUT_DIR / f"video_{uuid.uuid4().hex[:10]}.mp4"
    export_to_video(video_frames, str(out_path), fps=8)
    return str(out_path)


def generate_video_from_image(
    image_path: str,
    prompt: str,
    num_frames: int = 49,
    num_inference_steps: int = 50,
) -> str:
    """יוצר סרטון מתמונת פתיחה + תיאור טקסטואלי (image-to-video)."""
    from diffusers.utils import export_to_video, load_image

    pipe = _get_pipeline("i2v")
    image = load_image(image_path)
    result = pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    )
    video_frames = result.frames[0]

    out_path = OUTPUT_DIR / f"video_{uuid.uuid4().hex[:10]}.mp4"
    export_to_video(video_frames, str(out_path), fps=8)
    return str(out_path)
