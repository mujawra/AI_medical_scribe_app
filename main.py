from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import io
import base64
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
from datetime import datetime
from fpdf import FPDF

# Vercel serverless has no system ffmpeg — imageio_ffmpeg ships a static binary via pip.
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

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

def normalize_audio_to_wav(audio_bytes: bytes) -> bytes:
    """
    Phone browsers (esp. mobile Chrome/Safari) usually record in WebM/Opus or MP4/AAC,
    which speech_recognition's AudioFile cannot read directly (it needs WAV/AIFF/FLAC).
    This converts whatever format comes in into a clean 16kHz mono WAV using ffmpeg via pydub.
    Requires ffmpeg to be installed on the server.
    """
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as e:
        print(f"Audio normalization error: {e}")
        return audio_bytes  # fall back to original bytes if conversion fails

def transcribe_audio_hf(audio_bytes: bytes) -> str:
    # NOTE: api-inference.huggingface.co is deprecated (returns HTTP 410) as of 2026.
    # Hugging Face now routes all serverless inference through router.huggingface.co.
    API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=35)
        if response.status_code == 200:
            result = response.json()
            extracted_text = result.get("text", "").strip()
            hallucinations = ["Thank you for watching!", "Subtitles by", "Amara.org"]
            if any(h.lower() in extracted_text.lower() for h in hallucinations) and len(extracted_text.split()) < 4:
                return ""
            return extracted_text
    except Exception as e:
        print(f"HF Whisper Error: {e}")
    return ""

def transcribe_audio_fallback(audio_bytes: bytes) -> str:
    # 1. First Try Google Speech Recognition (Urdu & English) - Highly Accurate for Speech
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            # Urdu Try
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 1:
                return text.strip()
    except Exception as e:
        print(f"Urdu SR Error: {e}")

    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            # English Try
            text = recognizer.recognize_google(audio_data, language="en-US")
            if text and len(text.strip()) > 1:
                return text.strip()
    except Exception as e:
        print(f"English SR Error: {e}")

    # 2. Backup HF Whisper API
    text_hf = transcribe_audio_hf(audio_bytes)
    if text_hf and len(text_hf.strip()) > 1:
        return text_hf.strip()

    return ""

def transliterate_to_roman_urdu(urdu_text: str) -> str:
    """
    Converts Urdu-script text into Roman Urdu (same words, Latin letters) —
    a transliteration, not a translation. Falls back to the original text if it fails.
    """
    if not urdu_text or urdu_text.strip() == "":
        return urdu_text

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": "You convert Urdu-script text into readable Latin-letter text, preserving the original spoken meaning exactly. Two cases: (1) If the Urdu-script text represents actual Urdu words, transliterate them phonetically into Roman Urdu (the way Urdu speakers type on phones, e.g. 'mujhe bukhar hai'). (2) If the Urdu-script text is actually English words spelled out phonetically in Urdu letters (this happens when a speech recognizer mishears English speech as Urdu), recognize this and output the correct, properly-spelled English words instead (e.g. if the Urdu letters phonetically spell out 'patient is suffering from fever and headache', output exactly that in correct English spelling). Do NOT translate meaning between languages — only recover the correct written form of the same words that were spoken. Output ONLY the converted text, nothing else — no quotes, no explanation. If the input is already in Latin letters, return it unchanged."
        },
        {"role": "user", "content": urdu_text}
    ]
    payload_base = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 300
    }
    for model_id in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]:
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
                print(f"Roman Urdu transliteration HTTP {res.status_code} from {model_id}: {res.text[:200]}")
        except Exception as e:
            print(f"Roman Urdu transliteration error ({model_id}): {e}")
            continue

    return urdu_text  # fall back to original script if transliteration fails

def transcribe_long_audio(audio_bytes: bytes, chunk_seconds: int = 60) -> str:
    """
    Splits long audio into chunks (default 60s each), transcribes each chunk separately,
    and joins them into one full transcript. Needed because Google SR / HF Whisper are
    unreliable or timeout on long single files (e.g. a 30-minute recording).
    """
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        print(f"Long audio load error: {e}")
        return transcribe_audio_fallback(audio_bytes)  # fall back to single-shot on short/simple files

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
    """
    For long recordings: instead of showing the full verbatim transcript, ask the LLM to pull out
    only the medically relevant terms — symptoms, complaints, duration, history, anything spoken
    medicine names (as history) — as a short bullet list. Non-medical chit-chat is dropped.
    """
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
            "content": """You are reviewing a long spoken consultation transcript (possibly Urdu/Roman Urdu/English mixed).
Extract ONLY the medically relevant terms mentioned — symptoms, complaints, duration, severity, relevant history,
and any medicines the patient says they already took (list these as reported history only).
Ignore small talk, greetings, and anything not medically relevant.
Output ONLY a short bullet list of medical terms/phrases (translated to English), nothing else — no headers, no extra commentary.
If nothing medically relevant is found, output exactly: "No medical terms detected in this recording." """
        },
        {"role": "user", "content": full_transcript}
    ]
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500
    }
    try:
        res = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            result = res.json()
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0]["message"]["content"].strip()
                if output:
                    return output
    except Exception as e:
        print(f"Medical term extraction error: {e}")

    return "Medical term extraction failed — please review the full transcript manually."

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # If audio transcription was completely empty, don't call AI LLM
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
            "content": f"""You are an AI medical scribe assisting a doctor by turning a spoken consultation into structured clinical notes with a rough, non-final medication idea.

RULES (STRICT):
1. ONLY extract symptoms, complaints, history, or observations explicitly spoken in the audio transcript. Do not invent or assume anything not said.
2. Translate Urdu/Roman Urdu spoken text into professional English.
3. You may suggest ONE common, generic, low-risk over-the-counter medicine typically associated with the stated symptom (e.g. a general antipyretic/analgesic for fever/pain, an antacid for indigestion, ORS for dehydration) as a ROUGH IDEA ONLY, for simple/everyday symptoms only.
4. If you suggest a medicine, you may include its STANDARD ADULT TEXTBOOK REFERENCE DOSE (the generic range printed on any drug label, e.g. "Paracetamol 500-1000mg every 4-6 hours, max 4000mg/24h") — but you must label it clearly as a standard adult reference, not a personalized prescription, since it has not been adjusted for this specific patient's age, weight, allergies, or other conditions (none of which are knowable from voice alone).
5. If the transcript mentions anything serious or ambiguous (chest pain, breathing difficulty, severe/persistent symptoms, pregnancy, children, high fever, symptoms lasting many days, or anything you are not confident about), do NOT suggest any medicine or dose — write "Doctor must evaluate before any medication" instead.
6. If the patient mentions a medicine they already took, record it as history in Chief Complaint only — do not repeat or endorse it in the plan.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Translate spoken symptoms/history to English accurately, including any medicines the patient says they already took, reported as history only]
* **Possible Diagnosis:** [Primary differential(s) suggested by the spoken complaint, phrased as "to be confirmed by physician"]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention (Rough AI Idea — standard adult reference, NOT a personalized prescription):**
    * [Generic medicine name]: [standard adult reference dose, per Rule 4], OR "Doctor must evaluate before any medication" if Rule 5 applies.
* **⚠️ Disclaimer:** This is an AI-generated rough idea using a standard adult reference dose. It has not been adjusted for this patient's age, weight, allergies, or history, and must be reviewed and confirmed by the doctor before giving anything to the patient.
* **Advice/Next Steps:**
    * **Rest:** [General, non-drug guidance for this issue]
    * **Hydration:** [Relevant general fluid/dietary guidance]
    * **Monitor Symptoms:** [Key warning signs for this issue]
    * **Follow-up:** [Timeline for re-consultation with the doctor]"""
        },
        {
            "role": "user",
            "content": f'Audio Transcript: "{transcription_text}"'
        }
    ]

    models_to_try = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]

    for model_id in models_to_try:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.1,
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
        except Exception as err:
            print(f"Error calling {model_id}: {err}")
            continue

    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** {transcription_text}
* **Possible Diagnosis:** Evaluation required based on transcript.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention (Rough AI Idea — standard adult reference, NOT a personalized prescription):**
    * Doctor must evaluate before any medication.
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor before giving anything to the patient.
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
        audio_content = normalize_audio_to_wav(audio_content)

        # Check audio duration to decide short vs long processing path
        try:
            duration_seconds = len(AudioSegment.from_file(io.BytesIO(audio_content))) / 1000
        except Exception:
            duration_seconds = 0

        LONG_AUDIO_THRESHOLD_SECONDS = 120  # recordings longer than 2 minutes use chunked processing

        if duration_seconds > LONG_AUDIO_THRESHOLD_SECONDS:
            full_transcript = transcribe_long_audio(audio_content)
            transcribed_text = extract_medical_terms(full_transcript, doc_name, pat_name)
            transcript_section_title = "🎙️ Extracted Medical Terms (from long recording)"
        else:
            transcribed_text = transcribe_audio_fallback(audio_content)
            transcript_section_title = "🎙️ Voice Recording (Transcribed)"

        # Keep transcribed_text truly empty if nothing was recognized, so generate_medical_report
        # uses its own built-in "unclear audio" template instead of asking the LLM to interpret
        # a fake placeholder sentence (which was causing confused, free-form LLM replies).
        display_transcription = transcribed_text if transcribed_text else "Audio recorded but transcription was unclear."

        summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

        # Prepend the voice transcription as its own section above "Clinical Information"
        # so it always appears first in the app, regardless of frontend rendering order.
        summary_with_transcript = f"""### {transcript_section_title}

> {display_transcription}

---

{summary_text}"""

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
