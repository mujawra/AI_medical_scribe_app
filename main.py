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
    # 1. First Try Google Speech Recognition (Urdu Script)
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            # Urdu Try
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 1:
                return text.strip()  # Direct Urdu Script return
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

def transcribe_long_audio(audio_bytes: bytes, chunk_seconds: int = 60) -> str:
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
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
        "temperature": 0.0,
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
            "content": f"""You are an AI medical scribe. Strictly summarize ONLY what is stated in the provided Audio Transcript.

STRICT RULES:
1. For 'Chief Complaint', directly convert the Urdu/Phonetic Urdu transcript into a clear, direct English sentence of the EXACT complaint stated. DO NOT add words like 'travel', 'work', 'recent', or any external details NOT mentioned in the audio.
2. Only include symptoms explicitly stated.
3. Suggest ONE standard low-risk OTC medication strictly for simple symptoms (or write 'Doctor must evaluate' for complex cases).

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Direct, accurate English sentence matching ONLY the transcript text]
* **Possible Diagnosis:** [Primary differential suggested by the spoken complaint, phrased as 'to be confirmed by physician']

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
            "temperature": 0.0,
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

        # Check audio duration
        try:
            duration_seconds = len(AudioSegment.from_file(io.BytesIO(audio_content))) / 1000
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
