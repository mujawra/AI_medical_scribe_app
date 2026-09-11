from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import io
import base64
import tempfile
from typing import Optional
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
from datetime import datetime
from fpdf import FPDF

# Vercel serverless has no system ffmpeg/ffprobe — imageio_ffmpeg ships a static ffmpeg binary via pip.
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
# NOTE: there is no bundled ffprobe. We avoid needing it at all by always passing an
# explicit `format=` to AudioSegment.from_file() (see normalize_audio_to_wav below),
# which stops pydub from trying to auto-probe the file and failing silently on
# non-WAV formats (OGG, MP3, M4A, AAC, AMR, 3GP, etc. — common on Android).

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN", "")

latest_data = {
    "transcription": "No audio transcribed yet.", 
    "summary": "No report generated yet.", 
    "doctor": "Dr. Zainab", 
    "patient": "Patient", 
    "date": datetime.now().strftime("%Y-%m-%d")
}

@app.get("/")
def home():
    return {"status": "FastAPI Backend is Live on Vercel!"}

# Maps common file extensions (from the uploaded filename) and MIME content-types
# to the format string ffmpeg expects. Covers what browsers/Android record in.
EXTENSION_FORMAT_MAP = {
    "wav": "wav", "wave": "wav",
    "mp3": "mp3",
    "m4a": "m4a", "mp4": "mp4",
    "aac": "aac",
    "ogg": "ogg", "oga": "ogg", "opus": "ogg",
    "webm": "webm",
    "flac": "flac",
    "wma": "asf",
    "amr": "amr",
    "3gp": "3gp", "3gpp": "3gp",
}

CONTENT_TYPE_FORMAT_MAP = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/mp4": "mp4", "audio/x-m4a": "m4a", "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/ogg": "ogg", "audio/opus": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac", "audio/x-flac": "flac",
    "audio/x-ms-wma": "asf",
    "audio/amr": "amr", "audio/3gpp": "3gp",
}

def guess_audio_format(filename: str, content_type: str) -> Optional[str]:
    """
    Figures out the ffmpeg format name from the upload's filename extension first
    (most reliable), falling back to the browser/device-reported content-type.
    Returns None if we genuinely can't tell — pydub will then attempt auto-detection,
    which works for some formats but not others (see note above).
    """
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower().strip()
        if ext in EXTENSION_FORMAT_MAP:
            return EXTENSION_FORMAT_MAP[ext]

    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in CONTENT_TYPE_FORMAT_MAP:
            return CONTENT_TYPE_FORMAT_MAP[ct]

    return None

def loudness_normalize(audio_segment: AudioSegment) -> AudioSegment:
    """
    Boosts a quiet recording up to a healthy peak level. WhatsApp/Android voice notes
    (OGG/Opus) are frequently recorded at a much lower volume than typical WAV
    recordings, which makes speech recognition engines mishear them — this is a common
    real cause of 'garbled' transcripts that isn't about format/codec support at all,
    just perceived loudness. We target a peak of -1 dBFS, capping the boost so we don't
    amplify noise on an already-loud file.
    """
    try:
        change_needed = -1.0 - audio_segment.max_dBFS
        if change_needed > 0:  # only boost quiet audio, never reduce already-loud audio
            change_needed = min(change_needed, 25.0)  # cap extreme boosts (avoids blowing out noise)
            audio_segment = audio_segment.apply_gain(change_needed)
    except Exception as e:
        print(f"Loudness normalization skipped: {e}")
    return audio_segment

def normalize_audio_to_wav(audio_bytes: bytes, filename: str = "", content_type: str = "") -> bytes:
    """
    Converts whatever audio format comes in (WAV, MP3, M4A, AAC, OGG, WebM, FLAC,
    WMA, AMR, 3GP — covering laptop and Android recordings alike) into a clean,
    loudness-normalized 16kHz mono WAV, using ffmpeg directly via an explicit format
    hint so pydub never needs the missing ffprobe binary.
    """
    detected_format = guess_audio_format(filename, content_type)

    # Try the detected/likely format first, then fall back to a couple of common
    # alternates, then finally let pydub attempt full auto-detection as a last resort.
    candidate_formats = []
    if detected_format:
        candidate_formats.append(detected_format)
    for fmt in ["ogg", "webm", "mp3", "m4a", "wav", "aac", "3gp", "amr"]:
        if fmt not in candidate_formats:
            candidate_formats.append(fmt)

    last_error = None
    for fmt in candidate_formats:
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            audio_segment = loudness_normalize(audio_segment)
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            print(f"[NORMALIZE] Success with format={fmt!r} (filename={filename!r}, content_type={content_type!r})")
            return wav_io.getvalue()
        except Exception as e:
            last_error = e
            continue

    # Last resort: let pydub guess with no format hint at all.
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment = loudness_normalize(audio_segment)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        print(f"[NORMALIZE] Success with NO format hint (auto-detect) (filename={filename!r})")
        return wav_io.getvalue()
    except Exception as e:
        print(f"Audio normalization error (filename={filename!r}, content_type={content_type!r}, "
              f"detected_format={detected_format!r}): tried {candidate_formats}, "
              f"last error={last_error}, final error={e}")
        return audio_bytes  # fall back to original bytes if every attempt fails

def transcribe_audio_hf(audio_bytes: bytes) -> str:
    API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/wav",  # audio_bytes here is always our normalized WAV output
    }
    try:
        response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=35)
        if response.status_code == 200:
            result = response.json()
            extracted_text = result.get("text", "").strip()
            hallucinations = ["Thank you for watching!", "Subtitles by", "Amara.org"]
            if any(h.lower() in extracted_text.lower() for h in hallucinations) and len(extracted_text.split()) < 4:
                return ""
            return extracted_text
        else:
            print(f"HF Whisper HTTP {response.status_code}: {response.text[:300]}")
    except Exception as e:
        print(f"HF Whisper Error: {e}")
    return ""

def transcribe_audio_fallback(audio_bytes: bytes) -> str:
    # 1. Try HF Whisper FIRST — it auto-detects the spoken language (Urdu vs English vs
    #    mixed) and transcribes in that language's own native script, rather than us
    #    forcing a language guess. This avoids the garbled/mixed-up text that happened
    #    when Google's Urdu recognizer was forced onto English (or mixed) speech.
    text_hf = transcribe_audio_hf(audio_bytes)
    if text_hf and len(text_hf.strip()) > 1:
        print(f"[TRANSCRIBE] Used HF Whisper. Result: {text_hf.strip()[:200]}")
        return text_hf.strip()
    print("[TRANSCRIBE] HF Whisper returned nothing usable, falling back to Google SR.")

    # 2. Fallback: Google Speech Recognition, Urdu first
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 1:
                print(f"[TRANSCRIBE] Used Google SR (ur-PK) FALLBACK. Result: {text.strip()[:200]}")
                return text.strip()
    except Exception as e:
        print(f"Urdu SR Error: {e}")

    # 3. Fallback: Google Speech Recognition, English
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            if text and len(text.strip()) > 1:
                print(f"[TRANSCRIBE] Used Google SR (en-US) FALLBACK. Result: {text.strip()[:200]}")
                return text.strip()
    except Exception as e:
        print(f"English SR Error: {e}")

    return ""

def transcribe_long_audio(audio_bytes: bytes, chunk_seconds: int = 60) -> str:
    try:
        # audio_bytes here is already a normalized WAV (see process_audio), so no format hint needed.
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    except Exception as e:
        print(f"Long audio load error: {e}")
        return transcribe_audio_fallback(audio_bytes)

    total_ms = len(audio_segment)
    chunk_ms = chunk_seconds * 1000
    full_text_parts = []

    for start_ms in range(0, total_ms, chunk_ms):
        chunk = audio_segment[start_ms:start_ms + chunk_ms].set_channels(1).set_frame_rate(16000)
        chunk_io = io.BytesIO()
        chunk.export(chunk_io, format="wav")
        chunk_bytes = chunk_io.getvalue()

        chunk_text = transcribe_audio_fallback(chunk_bytes)
        if chunk_text:
            full_text_parts.append(chunk_text)

    return " ".join(full_text_parts).strip()

def extract_medical_terms(full_transcript: str, doctor_name: str, patient_name: str) -> str:
    if not full_transcript or full_transcript.strip() == "":
        return "No medical terms detected — audio was unclear or empty."

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": """You are reviewing a long spoken consultation transcript.
Extract ONLY the medically relevant terms explicitly mentioned.
Ignore small talk, greetings, and do NOT add extra details or assumptions.
Output ONLY a short bullet list of medical terms/phrases (translated to English), nothing else."""
        },
        {"role": "user", "content": full_transcript}
    ]
    payload_base = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500
    }
    for model_id in ["Qwen/Qwen2.5-7B-Instruct:fastest", "meta-llama/Llama-3.1-8B-Instruct:fastest"]:
        payload = {**payload_base, "model": model_id}
        try:
            res = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result["choices"]) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
            else:
                print(f"Medical term extraction HTTP {res.status_code} from {model_id}: {res.text[:300]}")
        except Exception as e:
            print(f"Medical term extraction error ({model_id}): {e}")
            continue

    return "Medical term extraction failed — please review the full transcript manually."

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    if not transcription_text or transcription_text.strip() == "":
        return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** Audio sound was unclear or empty.
* **Possible Diagnosis:** Please record the audio clearly again.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention (Rough AI Idea — standard adult reference, NOT a personalized prescription):**
    * Not applicable — no audio detected to process.
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor before giving anything to the patient.
* **Advice/Next Steps:**
    * **Rest:** Re-record speaking clearly into the microphone.
    * **Hydration:** N/A
    * **Monitor Symptoms:** N/A
    * **Follow-up:** Re-submit clear audio recording."""

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI medical scribe. Convert the spoken audio transcript into a clinical summary report.

STRICT PERSPECTIVE AND TRANSLATION RULES FOR 'Chief Complaint':
1. Convert any Urdu script, Roman Urdu, or phonetically written words into proper, correct English sentences.
2. ALWAYS use third-person phrasing such as "The patient is suffering from..." or "The patient reports...". NEVER use first-person "I have..." or "I am...".
3. If the transcript says phonetically "پیشنٹ اس سفرنگ فروم فیور اینڈ ہیڈک" (Patient is suffering from fever and headache), write EXACTLY: "The patient is suffering from fever and headache."
4. Do NOT add hallucinated details like travel, history, duration, or cause unless explicitly stated in the audio.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Third-person English sentence: e.g., "The patient is suffering from fever and headache."]
* **Possible Diagnosis:** [Primary differential suggested by the spoken complaint, phrased as "to be confirmed by physician"]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention (Rough AI Idea — standard adult reference, NOT a personalized prescription):**
    * [Generic medicine name or Doctor evaluation note]
* **⚠️ Disclaimer:** This is an AI-generated rough idea using a standard adult reference dose. It must be reviewed and confirmed by the doctor.
* **Advice/Next Steps:**
    * **Rest:** [General guidance]
    * **Hydration:** [Relevant fluid guidance]
    * **Monitor Symptoms:** [Key warning signs]
    * **Follow-up:** [Timeline for re-consultation]"""
        },
        {
            "role": "user",
            "content": f'Audio Transcript: "{transcription_text}"'
        }
    ]

    models_to_try = [
        "Qwen/Qwen2.5-7B-Instruct:fastest",
        "meta-llama/Llama-3.1-8B-Instruct:fastest"
    ]

    for model_id in models_to_try:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 800
        }
        try:
            res = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result["choices"]) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
            else:
                print(f"Report generation HTTP {res.status_code} from {model_id}: {res.text[:300]}")
        except Exception as err:
            print(f"Error calling {model_id}: {err}")
            continue

    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** The patient is suffering from fever and headache.
* **Possible Diagnosis:** Evaluation required based on transcript.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention (Rough AI Idea — standard adult reference, NOT a personalized prescription):**
    * Doctor must evaluate before any medication.
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor.
* **Advice/Next Steps:**
    * **Rest:** General rest advised.
    * **Hydration:** Maintain hydration.
    * **Monitor Symptoms:** Monitor condition.
    * **Follow-up:** Re-consult if needed."""

def clean_txt_for_pdf(text: str) -> str:
    return text.replace("**", "").replace("###", "").replace("📋", "").replace("🩺", "").replace("📝", "").encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf_bytes(summary_text, transcription_text, doc_name, pat_name, report_date) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_fill_color(26, 54, 93)
    pdf.rect(0, 0, 210, 32, 'F')
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "CLINICAL CONVERSATION REPORT", ln=True, align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    safe_summary = clean_txt_for_pdf(summary_text)
    pdf.multi_cell(0, 7, safe_summary.strip())
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, "Detected Audio Transcript:", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    safe_transcript = clean_txt_for_pdf(transcription_text)
    pdf.multi_cell(0, 5, f'"{safe_transcript}"')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_file.seek(0)
        pdf_bytes = tmp_file.read()
    
    if os.path.exists(tmp_file.name):
        os.remove(tmp_file.name)
        
    return pdf_bytes

def format_transcript_for_display(text: str) -> str:
    """
    Produces the final "Voice Recording (Transcribed)" text shown to the user.

    Rule: figure out the DOMINANT spoken language of the recording.
    - If the recording is primarily Urdu (even if a few English words like
      "pain" or "chest" are mixed in, which is normal code-switching in Pakistani
      clinics), the ENTIRE line is written in Urdu (Nastaliq) script — including
      those English words, spelled phonetically in Urdu script (e.g. "pain" -> "پین").
      Whatever script the transcript happened to arrive in, the output is always Urdu script.
    - If the recording is primarily English, it is left as clean English — not forced
      into Urdu.
    Always preserves the original words/meaning and their order; never invents,
    reorders, or "fixes" content. Falls back to the original raw text if the model
    call fails, so the user never sees an empty transcript.
    """
    if not text or not text.strip():
        return text

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are formatting a speech-to-text transcript for display. First, determine the "
                "DOMINANT spoken language of the transcript: Urdu, or English.\n\n"
                "CASE A — dominant language is Urdu (this includes Urdu sentences with a few English "
                "words mixed in, which is normal code-switching): rewrite the ENTIRE line in Urdu "
                "(Perso-Arabic/Nastaliq) script, no matter what script the input text is currently in. "
                "Any embedded English words (e.g. 'pain', 'chest', 'abdomen') must also be written "
                "phonetically in Urdu script, not left in Latin letters.\n\n"
                "CASE B — dominant language is English: output the text as clean, correctly spelled "
                "English, in Latin script.\n\n"
                "STRICT RULES (both cases): keep the exact same words and meaning, in the exact same "
                "order as given — do not reorder, rephrase, summarize, translate the meaning, or add "
                "anything not present. This is a script/spelling formatting pass only, not a rewrite. "
                "Output ONLY the final formatted text, nothing else — no quotes, no explanation, no "
                "case label."
            )
        },
        {"role": "user", "content": text}
    ]
    payload_base = {"messages": messages, "temperature": 0.2, "max_tokens": 400}
    for model_id in ["meta-llama/Llama-3.1-8B-Instruct:fastest", "Qwen/Qwen2.5-7B-Instruct:fastest"]:
        payload = {**payload_base, "model": model_id}
        try:
            res = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result["choices"]) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
            else:
                print(f"Transcript formatting HTTP {res.status_code} from {model_id}: {res.text[:300]}")
        except Exception as e:
            print(f"Transcript formatting error ({model_id}): {e}")
            continue

    return text  # fall back to the original raw text if every model call fails

@app.post("/process-audio")
@app.post("/process-audio/")
async def process_audio(
    audio: UploadFile = File(...), 
    doctor_name: str = Form("Dr. Zainab"), 
    patient_name: str = Form("Patient")
):
    global latest_data
    doc_name = doctor_name.strip() if doctor_name and doctor_name.strip() else "Dr. Zainab"
    pat_name = patient_name.strip() if patient_name and patient_name.strip() else "Patient"
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        audio_content = await audio.read()
        audio_content = normalize_audio_to_wav(
            audio_content,
            filename=audio.filename or "",
            content_type=audio.content_type or "",
        )

        # Check audio duration
        try:
            duration_seconds = len(AudioSegment.from_file(io.BytesIO(audio_content), format="wav")) / 1000
        except Exception:
            duration_seconds = 0

        LONG_AUDIO_THRESHOLD_SECONDS = 120

        if duration_seconds > LONG_AUDIO_THRESHOLD_SECONDS:
            full_transcript = transcribe_long_audio(audio_content)
            transcribed_text = extract_medical_terms(full_transcript, doc_name, pat_name)
            transcript_section_title = "🎙️ Voice Recording (Transcribed)"
        else:
            transcribed_text = transcribe_audio_fallback(audio_content)
            transcript_section_title = "🎙️ Voice Recording (Transcribed)"

        display_transcription = transcribed_text if transcribed_text else "Audio recorded but transcription was unclear."
        display_transcription = format_transcript_for_display(display_transcription)

        summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

        # Prepend top transcript block
        summary_with_transcript = f"### {transcript_section_title}\n\n> {display_transcription}\n\n---\n\n{summary_text}"

        pdf_bytes = generate_pdf_bytes(summary_text, display_transcription, doc_name, pat_name, current_date)
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        latest_data["transcription"] = display_transcription
        latest_data["summary"] = summary_text
        latest_data["doctor"] = doc_name
        latest_data["patient"] = pat_name
        latest_data["date"] = current_date

        return {
            "status": "success", 
            "transcription": display_transcription, 
            "summary": summary_with_transcript,
            "pdf_base64": pdf_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/download-pdf")
@app.get("/download-pdf/")
async def download_pdf():
    try:
        pdf_bytes = generate_pdf_bytes(
            latest_data["summary"], 
            latest_data["transcription"],
            latest_data["doctor"],
            latest_data["patient"],
            latest_data["date"]
        )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=Clinical_Report.pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF download error: {str(e)}")
