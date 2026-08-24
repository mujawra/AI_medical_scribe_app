from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import base64
import tempfile
from datetime import datetime
from fpdf import FPDF
from huggingface_hub import InferenceClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize HF Inference Client directly with your Token
client = InferenceClient(api_key=HF_TOKEN)

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

def transcribe_recording(audio_bytes: bytes) -> str:
    """Uses HF Inference Client for native audio stream transcription"""
    try:
        # Automatic whisper inference using HF Client SDK
        response = client.automatic_speech_recognition(
            audio=audio_bytes,
            model="openai/whisper-large-v3-turbo"
        )
        if isinstance(response, dict):
            text = response.get("text", "").strip()
        else:
            text = str(response).strip()
            
        # Hallucination Filter
        hallucinations = ["thank you for watching", "subtitles by", "amara.org"]
        if not any(h in text.lower() for h in hallucinations):
            return text

    except Exception as e:
        print(f"HF Audio Transcription Error: {e}")

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # Strictly handle empty or unreadable audio
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

    system_prompt = f"""You are an expert AI Medical Scribe assisting a physician.
You will receive a transcript of a spoken patient-doctor dialogue (spoken in Urdu, Roman Urdu, or English).

YOUR TASKS:
1. Extract exact symptoms strictly mentioned in the transcript.
2. Translate Urdu/Roman Urdu terms into clinical English terms.
3. Diagnose the condition based ONLY on the spoken content.
4. Dynamically generate appropriate medical prescriptions (generic drugs with dosages) and advice for the diagnosed condition.

OUTPUT FORMAT:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Translated spoken symptoms]
* **Possible Diagnosis:** [Primary Diagnosis]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Dynamically suggested generic medicine & dosage based on diagnosis]
* **Advice/Next Steps:**
    * **Rest:** [Relevant advice]
    * **Hydration:** [Relevant fluid/diet advice]
    * **Monitor Symptoms:** [Key warning signs]
    * **Follow-up:** [Timeline for re-consultation]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Audio Transcript: "{transcription_text}"'}
    ]

    try:
        # High quality open-source model through HF Router API
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as err:
        print(f"AI Report Generation Error: {err}")

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
        
        # 1. Transcribe direct audio stream via HF Hub Client SDK
        transcribed_text = transcribe_recording(audio_content)

        # 2. Send transcript directly to LLM for report/prescription generation
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
