"""
שירות 1: שיבוט קול / יצירת דיבור (Voice Cloning / TTS)
מבוסס על Coqui TTS (coqui-ai/TTS) - מודל xTTS v2

חשוב להבין איך שיבוט קול עובד בפועל:
xTTS לא "ממציא" קול מתוך תיאור מילולי כמו "סבתא מרוקאית זקנה".
הוא משכפל קול מתוך קובץ שמע לדוגמה (reference audio / speaker_wav) שאתה מספק -
בדרך כלל 6-30 שניות של הקול שאתה רוצה לשכפל.

אז כדי לקבל "סבתא מרוקאית זקנה" צריך אחד מהשניים:
1. קובץ שמע קצר של קול שכבר נשמע ככה (מוקלט, או שנוצר בכלי אחר) - והמודל ישכפל אותו.
2. ספריית "קולות מוכנים" (voice presets) שתכין מראש - זה מה שהוספתי כאן כ-VOICE_PRESETS,
   ואתה יכול למלא אותה בקבצי reference שאתה אוסף/מקליט מראש.

ה-`voice_description` שהמשתמש מקליד (למשל "סבתא מרוקאית זקנה") נשמר כמטא-דאטה/תיוג בלבד,
ומשמש לבחירת preset הכי קרוב מתוך הספרייה - הוא לא משפיע ישירות על הקול שנוצר.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

# ה-API האמיתי מתוך TTS-0.22.0/TTS/api.py:
#   from TTS.api import TTS
#   tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
#   tts.tts_to_file(text=..., speaker_wav="path/to/reference.wav", language="he", file_path="out.wav")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", "./outputs"))
VOICES_DIR = Path(os.environ.get("STUDIO_VOICES_DIR", "./voice_presets"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ספריית קולות מוכנים - למלא בקבצי reference audio (wav, 6-30 שניות, איכות טובה, ללא רעש רקע)
# שם הקובץ = מזהה ה-preset. אפשר להוסיף עוד רשומות כאן, או לטעון דינמית מהתיקייה.
VOICE_PRESETS = {
    "moroccan_grandma": VOICES_DIR / "moroccan_grandma.wav",
    "young_male_energetic": VOICES_DIR / "young_male_energetic.wav",
    "formal_news_anchor": VOICES_DIR / "formal_news_anchor.wav",
}

_tts_model = None  # נטען פעם אחת (lazy load) - המודל כבד וטעינה חוזרת מיותרת


def _get_model():
    """טוען את מודל ה-TTS פעם אחת בלבד (singleton) כדי לא לטעון GB מודל בכל בקשה."""
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS  # ייבוא מתעכב - כדי שהמודול הזה ייטען מהר גם בלי הספרייה מותקנת
        _tts_model = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=True)
    return _tts_model


def list_voice_presets() -> list[str]:
    """מחזיר את שמות הקולות המוכנים הזמינים (רק אלה שיש להם קובץ בפועל)."""
    return [name for name, path in VOICE_PRESETS.items() if path.exists()]


def synthesize_speech(
    text: str,
    language: str = "he",
    speaker_wav_path: Optional[str] = None,
    voice_preset: Optional[str] = None,
    voice_description: Optional[str] = None,  # נשמר לתיוג/לוגים בלבד, לא משפיע על הסינתזה
) -> str:
    """
    יוצר קובץ שמע מדובר מטקסט, בקול משוכפל.

    Args:
        text: הטקסט שיוקרא
        language: קוד שפה (he אינו נתמך רשמית ב-xTTS v2 הבסיסי - ראו הערה למטה)
        speaker_wav_path: נתיב לקובץ שמע reference שהמשתמש העלה (עדיפות ראשונה)
        voice_preset: מזהה preset מתוך VOICE_PRESETS (אם לא הועלה reference משלו)
        voice_description: טקסט חופשי כמו "סבתא מרוקאית זקנה" - תיוג בלבד

    Returns:
        נתיב לקובץ ה-wav שנוצר
    """
    if not speaker_wav_path and not voice_preset:
        raise ValueError(
            "צריך לספק reference audio (speaker_wav_path) או voice_preset קיים. "
            "תיאור מילולי בלבד (voice_description) אינו מספיק ליצירת הקול."
        )

    ref_path = speaker_wav_path or str(VOICE_PRESETS.get(voice_preset, ""))
    if not ref_path or not Path(ref_path).exists():
        raise FileNotFoundError(f"קובץ קול הייחוס לא נמצא: {ref_path}")

    # הערה חשובה: xTTS v2 תומך רשמית ב-17 שפות (אנגלית, ספרדית, צרפתית, ערבית ועוד) -
    # עברית אינה ברשימה הרשמית נכון לגרסה זו. יש לבדוק את tts.languages בזמן ריצה,
    # ואם נדרשת עברית איכותית - כדאי לשלב מודל TTS עברי ייעודי (למשל דרך Hugging Face)
    # במקום xTTS, או לתרגם/לתמלל בעקיפין.
    model = _get_model()
    out_name = f"speech_{uuid.uuid4().hex[:10]}.wav"
    out_path = OUTPUT_DIR / out_name

    model.tts_to_file(
        text=text,
        speaker_wav=ref_path,
        language=language,
        file_path=str(out_path),
    )
    return str(out_path)


if __name__ == "__main__":
    # בדיקה ידנית מקומית (דורש GPU + המודל מותקן + קובץ reference אמיתי)
    print("Available presets:", list_voice_presets())
