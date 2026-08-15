from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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

def transcribe_audio_hf(audio_bytes: bytes) -> str:
    """Accurate Hugging Face Whisper API transcription for Urdu and English mobile voice recordings"""
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "").strip()
    except Exception as e:
        print(f"HF Whisper Error: {e}")
    return ""

def transcribe_audio_fallback(audio_bytes: bytes) -> str:
    text = transcribe_audio_hf(audio_bytes)
    if text and len(text.strip()) > 2:
        return text.strip()

    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 2:
                return text.strip()
    except Exception as e:
        print(f"Urdu SpeechRecognition Warning: {e}")

    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            if text and len(text.strip()) > 2:
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

    messages = [
        {
            "role": "system",
            "content": f"""You are an expert AI Medical Scribe.
Analyze the audio transcript and generate a professional clinical medical report strictly matching Pic 2 layout.

LANGUAGE INSTRUCTION:
- If the spoken audio transcript is in Urdu, Roman Urdu, or Hindi, TRANSLATE IT AND WRITE THE ENTIRE REPORT EXCLUSIVELY IN CLEAR PROFESSIONAL ENGLISH.

DYNAMIC CLINICAL INSTRUCTIONS:
1. Extract exact complaints, symptoms, and duration strictly from the audio transcript (translate Urdu to English).
2. Deduce possible diagnosis based ONLY on the spoken symptoms.
3. Recommend standard over-the-counter medicine with dosage matching the detected condition.
4. DO NOT use generic bracket labels like [Condition-Specific Precaution]. Strictly use the exact sub-headings as formatted below.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Translate spoken complaints to English and write detailed symptoms here]
* **Possible Diagnosis:** [Clinical condition deduced dynamically from transcript]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Medication Name (Generic Name)]: [Dosage, frequency, and duration tailored to condition]
* **Advice/Next Steps:**
    * **Rest:** [Rest advice tailored to condition]
    * **Hydration:** [Fluid intake guidance tailored to condition]
    * **Monitor Symptoms:** [Symptom monitoring and warning sign instructions]
    * **Follow-up:** [Follow-up appointment time frame]"""
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

* **Chief Complaint:** Patient reported experiencing symptoms mentioned in the clinical consultation audio.
* **Possible Diagnosis:** Clinical evaluation required based on audio transcript.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * Symptomatic treatment as recommended by physician.
* **Advice/Next Steps:**
    * **Rest:** Ensure adequate rest to aid recovery.
    * **Hydration:** Maintain good daily fluid intake.
    * **Monitor Symptoms:** Observe for persistent or worsening symptoms.
    * **Follow-up:** Consult doctor if condition does not improve in 2-3 days."""

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
    
    return pdf.output(dest='S').encode('latin-1')

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
        transcribed_text = "Audio recorded but text transcription could not clear noise. Patient requested symptom review."

    summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

    latest_data["transcription"] = transcribed_text
    latest_data["summary"] = summary_text
    latest_data["doctor"] = doc_name
    latest_data["patient"] = pat_name
    latest_data["date"] = current_date

    return {
        "status": "success", 
        "transcription": transcribed_text, 
        "summary": summary_text
    }

@app.get("/download-pdf")
@app.get("/download-pdf/")
async def download_pdf():
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
