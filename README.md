# מעבדת מדיה — Voice Cloning + Lip-Sync + Video Generation

מבוסס על שלושה פרויקטים בקוד פתוח שסיפקת:
- **coqui-ai/TTS** (xTTS v2) — שיבוט קול
- **Rudrabha/Wav2Lip** — סנכרון תנועות שפתיים
- **zai-org/CogVideo** (CogVideoX) — יצירת סרטון מטקסט/תמונה

## מבנה הפרויקט

```
studio/
  backend/
    main.py                          # FastAPI, 5 endpoints
    requirements.txt
    services/
      tts_service.py                 # שירות 1: קול
      lipsync_service.py             # שירות 2: שפתיים
      videogen_service.py            # שירות 3: וידאו
      pipeline_full.py               # שירות 4: הכל ביחד
      pipeline_voice_lipsync.py      # שירות 5: קול+שפתיים (הבקשה המקורית שלך)
  frontend/
    index.html                       # ממשק אחד, 5 טאבים
```

## ⚠️ לפני הכל: דרישות חומרה אמיתיות

זו הנקודה הכי חשובה להבין לפני שמתחילים:

| שירות | דרישת GPU | גודל מודל | זמן טעינה ראשוני |
|---|---|---|---|
| TTS (xTTS v2) | מומלץ, לא חובה | ~2GB | דקה-שתיים |
| Wav2Lip | מומלץ | ~350MB | מהיר |
| CogVideoX-5b | **חובה בפועל**, ~16-24GB VRAM | **~20-30GB** | הורדה ארוכה בפעם הראשונה |

**המשמעות המעשית:** אם המחשב שלך הוא לפטופ/מחשב בית רגיל בלי כרטיס NVIDIA חזק (RTX 3090/4090 ומעלה, או שקול), שירות יצירת הווידאו (CogVideo) כנראה **לא יעבוד בסבירות טובה**, ואילו TTS ו-Wav2Lip יעבדו, רק לאט יותר על CPU.

**המלצה מעשית:** התחל עם שירותים 1, 2, 5 (קול, שפתיים, קול+שפתיים משולב) — הם כבדים הרבה פחות ומתאימים בדיוק לתרחיש שתיארת בהתחלה (סרטון AI קיים + טקסט + הגדרת קול). את שירותים 3 ו-4 (יצירת וידאו) כדאי להריץ רק אם יש GPU חזק, או לשכור GPU בענן (RunPod / Vast.ai / AWS) לפי הצורך, בתשלום לפי שעה.

## ⚠️ Vercel — למה זה לא יכול לארח את כל המערכת

Vercel הוא **serverless**: הפונקציות שלו נטענות לפי בקשה, יש להן timeout קצר (שניות עד דקות בודדות בתלות בתוכנית), ואין להן GPU ואין מקום לאחסן מודלים ששוקלים ג'יגה-בייטים. עיבוד וידאו/קול עם המודלים האלה לוקח דקות ודורש GPU רציף.

**מה כן אפשר על Vercel:** לארח את `frontend/index.html` (או גרסת Next.js שלו) כממשק. הוא רק שולח בקשות HTTP ל-backend.

**מה חייב לרוץ במקום אחר (עם GPU):**
- מקומית, על המחשב/שרת שלך (אם יש GPU מתאים)
- או שרת ענן שכור לפי שעה — RunPod, Vast.ai, Lambda Labs, AWS EC2 עם GPU

כדי לחבר frontend ב-Vercel ל-backend חיצוני: משנים את `API_BASE` בראש `index.html` לכתובת הציבורית של ה-backend (למשל `https://your-runpod-instance.proxy.runpod.net`), ומוודאים ש-CORS פתוח (כבר מוגדר כך ב-`main.py`).

## התקנה מקומית (על מכונה עם GPU)

```bash
cd backend
python -m venv venv
source venv/bin/activate   # ב-Windows: venv\Scripts\activate

# חשוב: להתקין torch עם CUDA המתאים לכרטיס שלך לפני שאר החבילות
# ראו: https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### הכנת קבצים נדרשים (לא כלולים ב-repo, צריך להוריד בנפרד)

1. **Wav2Lip checkpoint** — הורד `wav2lip_gan.pth` (מקושר ב-README המקורי של Wav2Lip) ושים ב-
   `backend/Wav2Lip-master/checkpoints/wav2lip_gan.pth`
2. **קבצי reference לשיבוט קול** — הכן קבצי wav קצרים (6-30 שניות, איכות טובה, בלי רעש רקע)
   ושים אותם ב-`backend/voice_presets/`, למשל:
   - `moroccan_grandma.wav`
   - `young_male_energetic.wav`

   **הערה חשובה:** xTTS משכפל קול מתוך קובץ שמע אמיתי — הוא לא "ממציא" קול מתיאור מילולי.
   כדי לקבל "סבתא מרוקאית זקנה" צריך *דוגמת הקלטה אמיתית* של קול כזה (מוקלטת או שנאספה),
   לא רק את הטקסט "סבתא מרוקאית זקנה". השדה `voice_description` בממשק הוא לתיוג בלבד.
3. **CogVideoX** — יורד אוטומטית מ-Hugging Face בריצה הראשונה (דורש אינטרנט ו-~25GB פנויים).

### הרצה

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

ואז פותחים את `frontend/index.html` בדפדפן (או מגישים אותו דרך שרת סטטי כלשהו / Vercel).

## שימוש מהיר (בלי frontend, ישירות מ-Python)

התרחיש שתיארת בהתחלה — סרטון AI קיים + טקסט + הגדרת קול — זה בדיוק `pipeline_voice_lipsync`:

```python
from services import pipeline_voice_lipsync

result = pipeline_voice_lipsync.run(
    video_path="my_ai_video.mp4",
    text="שלום, איך שלומך היום?",
    language="he",
    speaker_wav_path="voice_presets/moroccan_grandma.wav",  # קובץ reference אמיתי
)
print(result["final_video_path"])
```

## מגבלות ידועות שכדאי להכיר

- **עברית ב-xTTS**: xTTS v2 תומך רשמית ב-17 שפות ועברית אינה ביניהן. יש לבדוק בפועל
  (`tts.languages`) — יתכן שהתוצאה באיכות נמוכה יותר מאשר בשפות הנתמכות, או שיידרש מודל עברי ייעודי.
- **Wav2Lip** נותן תוצאות טובות בעיקר על פנים קרובות/מרכזיות במסך, בתאורה טובה — לא על כל סרטון.
- כל השירותים כאן משתמשים בקוד המקור בדיוק כפי שהעלית (ללא שינויים ב-repos עצמם) —
  העטיפה (services/) קוראת להם כמו שהם, לפי ה-API/CLI המתועד בכל אחד.
