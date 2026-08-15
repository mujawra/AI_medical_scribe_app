from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import io
import base64
import tempfile
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
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
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
    text = transcribe_audio_hf(audio_bytes)
    if text and len(text.strip()) > 3:
        return text.strip()

    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
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
            "content": f"""You are a strict AI Medical Assistant / Clinical Scribe helping a Doctor.

STRICT LAWS YOU MUST FOLLOW:
1. STRICT GROUNDING: Extract ONLY what is explicitly mentioned in the audio transcript. 
2. NO HALLUCINATIONS: Do NOT invent extra symptoms (e.g. if the patient says "thigh pain", DO NOT write fever, headache, or chest pain).
3. DOCTOR ASSISTANT ROLE: Provide preliminary notes and initial supportive advice for the doctor to review and finalize.
4. TRANSLATION: If input is in Urdu or Roman Urdu, translate the EXACT spoken symptoms into clear English.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [EXACT English translation of symptoms spoken in audio ONLY]
* **Possible Diagnosis:** [Primary differential diagnosis for the exact spoken complaint ONLY]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Targeted OTC/Generic recommendation strictly for the stated issue, or 'For doctor to prescribe upon evaluation']
* **Advice/Next Steps:**
    * **Rest:** [Targeted advice for the specific body area/symptom spoken]
    * **Hydration/Diet:** [Relevant supportive care]
    * **Monitor Symptoms:** [Key warning signs specific ONLY to the reported issue]
    * **Follow-up:** [Timeline for re-consultation]"""
        },
        {
            "role": "user",
            "content": f'Spoken Audio Transcript: "{transcription_text}"'
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
* **Possible Diagnosis:** Direct evaluation required by doctor.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * To be determined by attending physician.
* **Advice/Next Steps:**
    * **Rest:** General rest advised.
    * **Hydration/Diet:** Normal diet.
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
        transcribed_text = transcribe_audio_fallback(audio_content)

        if not transcribed_text:
            transcribed_text = "Audio recorded but transcription was unclear. Patient requested clinical review."

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
