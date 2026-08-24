from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import base64
import tempfile
from datetime import datetime
from fpdf import FPDF

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

latest_data = {
    "transcription": "", 
    "summary": "", 
    "doctor": "Dr. Zainab", 
    "patient": "Patient", 
    "date": datetime.now().strftime("%Y-%m-%d")
}

@app.get("/")
def home():
    return {"status": "FastAPI Medical Scribe Engine Live"}

def transcribe_recording(audio_bytes: bytes, filename: str) -> str:
    """Accurately convert Urdu/Roman-Urdu/English mobile audio to text"""
    
    # 1. Best Choice: Groq API (High Speed & Handles Mobile formats directly)
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            files = {"file": (filename or "audio.m4a", audio_bytes)}
            data = {"model": "whisper-large-v3"}
            
            res = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            if res.status_code == 200:
                text = res.json().get("text", "").strip()
                if text:
                    return text
        except Exception as e:
            print(f"Groq Transcription Error: {e}")

    # 2. Backup: HuggingFace Inference API
    try:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        res = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=35)
        if res.status_code == 200:
            result = res.json()
            if isinstance(result, dict) and "text" in result:
                text = result.get("text", "").strip()
                # Remove common Whisper silent hallucinations
                hallucinations = ["thank you for watching", "subtitles by", "amara.org", "you"]
                if not any(h in text.lower() for h in hallucinations):
                    return text
    except Exception as e:
        print(f"HF Whisper Error: {e}")

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # Strictly handle empty or unreadable audio without fake hardcoded reports
    if not transcription_text or len(transcription_text.strip()) < 3:
        return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** Could not extract clear speech from audio.
* **Possible Diagnosis:** Pending clear voice input.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * No audio detected to prescribe medication.
* **Advice/Next Steps:**
    * **Rest:** Please re-record speaking clearly into the mic.
    * **Hydration:** N/A
    * **Monitor Symptoms:** Re-record audio.
    * **Follow-up:** Re-submit recording."""

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Pure AI instruction prompt without hardcoded drug names
    system_prompt = f"""You are an expert AI Medical Scribe assisting a physician.
You will receive a transcript of a spoken patient-doctor dialogue (spoken in Urdu, Roman Urdu, or English).

YOUR TASKS:
1. Extract exact symptoms and complaints strictly mentioned in the transcript.
2. Translate Urdu/Roman Urdu terms into clinical English terms.
3. Diagnose the condition based ONLY on the spoken content.
4. Dynamically generate appropriate medical prescriptions (generic drugs with dosages) and advice for the diagnosed condition.

OUTPUT FORMAT (Strictly follow this layout):

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Translated spoken symptoms]
* **Possible Diagnosis:** [Primary Diagnosis]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Dynamically suggested medicine & dosage based on diagnosis]
* **Advice/Next Steps:**
    * **Rest:** [Relevant advice]
    * **Hydration:** [Relevant fluid/diet advice]
    * **Monitor Symptoms:** [Key warning signs]
    * **Follow-up:** [Timeline for re-consultation]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Audio Transcript: "{transcription_text}"'}
    ]

    models_to_try = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]

    for model_id in models_to_try:
        try:
            res = requests.post(
                ROUTER_URL, 
                headers=headers, 
                json={"model": model_id, "messages": messages, "temperature": 0.1, "max_tokens": 800}, 
                timeout=25
            )
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
        except Exception as err:
            print(f"Error calling {model_id}: {err}")
            continue

    return "Failed to process AI clinical response."

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
    pdf.multi_cell(0, 7, clean_txt_for_pdf(summary_text).strip())
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, "Detected Audio Transcript:", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, f'"{clean_txt_for_pdf(transcription_text)}"')
    
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
        filename = audio.filename if audio.filename else "recording.m4a"
        
        # Real transcription check
        transcribed_text = transcribe_recording(audio_content, filename)

        # Send extracted transcription directly to AI Model
        summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

        pdf_bytes = generate_pdf_bytes(summary_text, transcribed_text, doc_name, pat_name, current_date)
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        latest_data["transcription"] = transcribed_text
        latest_data["summary"] = summary_text
        latest_data["doctor"] = doc_name
        latest_data["patient"] = pat_name
        latest_data["date"] = current_date

        return {
            "status": "success", 
            "transcription": transcribed_text, 
            "summary": summary_text,
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
