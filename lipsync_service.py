"""
שירות 2: תיקון/סנכרון תנועות שפתיים (Lip-Sync)
מבוסס על Rudrabha/Wav2Lip

Wav2Lip הוא סקריפט CLI (inference.py) ולא ספריית Python נקייה, אז העטיפה כאן
מריצה אותו כתת-תהליך (subprocess). זה בדיוק איך שמריצים אותו גם ב-README המקורי:

    python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <audio.wav> \
        --outfile <result.mp4>

חשוב: צריך checkpoint מאומן מראש (wav2lip_gan.pth) שלא כלול ב-repo עצמו -
יש להוריד אותו בנפרד (מקושר ב-README של הפרויקט) ולשים בתיקיית checkpoints/.
"""

import os
import subprocess
import uuid
from pathlib import Path

WAV2LIP_DIR = Path(os.environ.get("WAV2LIP_DIR", "./Wav2Lip-master"))
CHECKPOINT_PATH = Path(os.environ.get(
    "WAV2LIP_CHECKPOINT", str(WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth")
))
OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_lipsync(video_path: str, audio_path: str, resize_factor: int = 1) -> str:
    """
    מסנכרן תנועות שפתיים בסרטון קיים לפי קובץ שמע חדש.

    Args:
        video_path: נתיב לסרטון המקור (הפנים שרוצים להזיז)
        audio_path: נתיב לקובץ השמע החדש (מה שהשפתיים יסונכרנו אליו)
        resize_factor: הקטנת רזולוציה לפני עיבוד (1=מקורי, 2=חצי) - מאיץ עיבוד על GPU חלש

    Returns:
        נתיב לסרטון הפלט המסונכרן
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"לא נמצא checkpoint של Wav2Lip ב-{CHECKPOINT_PATH}. "
            "יש להוריד את wav2lip_gan.pth (מקושר ב-README של Wav2Lip) "
            "ולשים בתיקיית checkpoints/ - זה לא כלול ב-repo עצמו."
        )
    if not Path(video_path).exists():
        raise FileNotFoundError(f"לא נמצא קובץ הווידאו: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"לא נמצא קובץ השמע: {audio_path}")

    out_name = f"lipsync_{uuid.uuid4().hex[:10]}.mp4"
    out_path = OUTPUT_DIR / out_name

    cmd = [
        "python", str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(CHECKPOINT_PATH),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(out_path.resolve()),
        "--resize_factor", str(resize_factor),
    ]

    result = subprocess.run(
        cmd, cwd=str(WAV2LIP_DIR), capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip נכשל:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    if not out_path.exists():
        raise RuntimeError("Wav2Lip רץ בהצלחה אבל קובץ הפלט לא נוצר - בדוק לוגים.")

    return str(out_path)


if __name__ == "__main__":
    print("Checkpoint exists:", CHECKPOINT_PATH.exists())
