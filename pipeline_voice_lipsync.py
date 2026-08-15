"""
שירות 5: שילוב שיבוט קול + סנכרון שפתיים (ללא יצירת וידאו)
זה תואם בדיוק לבקשה המקורית שלך: יש כבר סרטון AI מוכן + טקסט + הגדרת קול -
והפלט הוא אותו סרטון עם קול חדש ושפתיים מסונכרנות.

שלבים:
  1. TTS (tts_service) - יוצר קובץ שמע מהטקסט, בקול הרצוי
  2. Wav2Lip (lipsync_service) - מסנכרן את השפתיים בסרטון הקיים לקול החדש
"""

from pathlib import Path
from typing import Optional

from . import tts_service
from . import lipsync_service


def run(
    video_path: str,
    text: str,
    language: str = "he",
    speaker_wav_path: Optional[str] = None,
    voice_preset: Optional[str] = None,
    voice_description: Optional[str] = None,
) -> dict:
    """
    Args:
        video_path: נתיב לסרטון ה-AI הקיים שהמשתמש העלה
        text: הטקסט שהדמות תגיד
        language: קוד שפה לסינתזה
        speaker_wav_path: קובץ reference שהמשתמש העלה לשיבוט קול (עדיפות)
        voice_preset: או - שם preset מוכן מהספרייה (למשל "moroccan_grandma")
        voice_description: תיאור חופשי (רק לתיוג/לוגים, לא משפיע על הקול בפועל)

    Returns:
        dict עם נתיבי קבצי הביניים והפלט הסופי
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"לא נמצא קובץ הווידאו: {video_path}")

    # שלב 1: יצירת קול
    audio_path = tts_service.synthesize_speech(
        text=text,
        language=language,
        speaker_wav_path=speaker_wav_path,
        voice_preset=voice_preset,
        voice_description=voice_description,
    )

    # שלב 2: סנכרון שפתיים לקול החדש
    final_video_path = lipsync_service.apply_lipsync(
        video_path=video_path,
        audio_path=audio_path,
    )

    return {
        "audio_path": audio_path,
        "final_video_path": final_video_path,
    }
