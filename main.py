from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
import io
import speech_recognition as sr
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

latest_data = {"transcription": "", "summary": "", "doctor": "", "patient": "", "date": ""}

@app.get("/")
@app.get("/process-audio")
def home():
    return {"status": "FastAPI Backend is Live on Vercel!"}

def transcribe_audio_fallback(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    
    # Try Urdu SpeechRecognition
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 3:
                return text.strip()
    except Exception as e:
        print(f"Urdu SpeechRecognition Warning: {e}")

    # Try English SpeechRecognition
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            if text and len(text.strip()) > 3:
                return text.strip()
    except Exception as e:
        print(f"English SpeechRecognition Warning: {e}")

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Dynamic Audio-Based Deductions Prompt (No Hardcoded Diseases/Meds)
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert AI Medical Scribe.
Analyze the provided audio transcript and generate a structured clinical report strictly matching the layout of Pic 2.

DYNAMIC CLINICAL INSTRUCTIONS:
1. Extract exact complaints, symptoms, and timeline explicitly mentioned in the audio.
2. Deduce the Possible Diagnosis based ONLY on the actual symptoms described in the transcript.
3. PRECAUTIONS & ADVICE MUST BE 100% SPECIFIC TO THE DETECTED CONDITION:
   - Do NOT use hardcoded generic advice or fixed templates.
   - Dynamically tailor all advice, precautions, and follow-ups strictly to the exact illness/symptoms found in the transcript.
4. For 'Suggested Medication/Intervention', list medications ONLY if explicitly spoken in the audio. If no medications were spoken, state "No specific medication mentioned in audio".

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Detailed description of symptoms extracted strictly from the audio]
* **Possible Diagnosis:** [Clinical impression deduced strictly from audio symptoms]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Explicitly mentioned medications with dosage if spoken, or "No specific medication mentioned in audio"]
* **Advice/Next Steps:**
    * **[Condition-Specific Precaution 1]:** [Detailed precaution tailored strictly to the detected illness]
    * **[Condition-Specific Precaution 2]:** [Second precaution tailored strictly to the detected illness]
    * **[Condition-Specific Precaution 3]:** [Targeted follow-up instruction relevant to this condition]"""
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

    # Fallback Structure
    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** {transcription_text}
* **Possible Diagnosis:** Clinical evaluation required based on spoken symptoms.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * No specific medication mentioned in audio.
* **Advice/Next Steps:**
    * **Targeted Care:** Follow condition-specific care guidance as recommended by physician.
    * **Monitoring:** Observe symptoms closely and seek re-evaluation if condition worsens."""

def clean_txt_for_pdf(text: str) -> str:
    return text.replace("**", "").replace("###", "").replace("📋", "").replace("🩺", "").replace("📝", "").encode('latin-1', 'ignore').decode('latin-1')

def generate_robust_pdf(filename, summary_text, transcription_text, doc_name, pat_name, report_date):
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
    
    pdf.output(filename)

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

    audio_content = await audio.read()
    
    transcribed_text = transcribe_audio_fallback(audio_content)

    if not transcribed_text:
        transcribed_text = "Patient reported symptoms during voice consultation."

    summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

    latest_data["transcription"] = transcribed_text
    latest_data["summary"] = summary_text
    latest_data["doctor"] = doc_name
    latest_data["patient"] = pat_name
    latest_data["date"] = current_date

    return {"status": "success", "transcription": transcribed_text, "summary": summary_text}

@app.get("/download-pdf")
@app.get("/download-pdf/")
async def download_pdf():
    pdf_filename = "/tmp/Clinical_Report.pdf"
    generate_robust_pdf(
        pdf_filename, 
        latest_data["summary"], 
        latest_data["transcription"],
        latest_data["doctor"],
        latest_data["patient"],
        latest_data["date"]
    )
    if os.path.exists(pdf_filename):
        return FileResponse(pdf_filename, media_type="application/pdf", filename="Clinical_Report.pdf")
    return {"status": "error", "message": "PDF generation failed."}
