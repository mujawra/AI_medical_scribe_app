from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import requests
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

def transcribe_audio_hf(audio_bytes: bytes, filename: str) -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("CRITICAL ERROR: HF_TOKEN missing!")
        return ""

    models = [
        "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
        "https://api-inference.huggingface.co/models/openai/whisper-large-v3",
        "https://api-inference.huggingface.co/models/openai/whisper-small"
    ]

    # Strategy 1: Send as Multipart Form Data (Fixes HF binary upload rejection)
    headers = {"Authorization": f"Bearer {token}"}
    params = {"wait_for_model": "true"}

    for api_url in models:
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            response = requests.post(api_url, headers=headers, files=files, params=params, timeout=40)
            print(f"HF Model: {api_url.split('/')[-1]} | Status: {response.status_code}")

            if response.status_code == 200:
                res = response.json()
                text = ""
                if isinstance(res, dict) and "text" in res:
                    text = res["text"].strip()
                elif isinstance(res, list) and len(res) > 0 and "text" in res[0]:
                    text = res[0]["text"].strip()
                
                if text and len(text) > 1:
                    return text
            
            # Strategy 2: Fallback to Raw Binary Stream if multipart fails
            headers_raw = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}
            res_raw = requests.post(api_url, headers=headers_raw, data=audio_bytes, params=params, timeout=35)
            if res_raw.status_code == 200:
                data = res_raw.json()
                if isinstance(data, dict) and "text" in data:
                    return data["text"].strip()
                elif isinstance(data, list) and len(data) > 0 and "text" in data[0]:
                    return data[0]["text"].strip()

        except Exception as e:
            print(f"Exception on {api_url}: {e}")
            continue

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Dynamic Prompt: NO hardcoded medicines inside prompt.
    # The AI dynamically detects illness & suggests appropriate medical intervention based on the audio transcript.
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert AI Medical Scribe.
Analyze the audio transcript of the patient-doctor interaction and generate a complete clinical report.

DYNAMIC ANALYSIS GUIDELINES:
1. Extract patient complaints, symptoms, and duration strictly from what was spoken in the transcript.
2. Based ON THE EXTRACTED SYMPTOMS AND ILLNESS, dynamically suggest appropriate standard medications, medical interventions, or over-the-counter care suitable for those specific symptoms. DO NOT use hardcoded examples—deduce directly from the patient's condition described in the transcript.
3. Transliterate any Roman Urdu / Hindi medical terms accurately into clear clinical English.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Detailed description of symptoms, complaints, and timeline extracted from transcript]
* **Possible Diagnosis:** [Clinical impression/diagnosis deduced from transcript]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:** [Dynamically suggested medications, dosages, or treatments appropriate for the detected condition]
* **Advice/Next Steps:** [General care, lifestyle/dietary guidance, and follow-up timeline]"""
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
            "temperature": 0.2,
            "max_tokens": 700
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

    # Fallback response
    return f"### 📋 Clinical Information\n\n* **Doctor Name:** {doctor_name}\n* **Patient Name:** {patient_name}\n* **Date:** {report_date}\n\n### 🩺 Medical Summary Report\n\n* **Chief Complaint:** {transcription_text}\n* **Possible Diagnosis:** Symptomatic evaluation required.\n\n### 📝 Recommended Prescription & Plan\n\n* **Suggested Medication/Intervention:** Consult physician for specific dosage and prescription.\n* **Advice/Next Steps:** Rest, proper hydration, and follow-up checkup as needed."

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
    doc_name = doctor_name.strip() if doctor_name and doctor_name.strip() else "Doctor"
    pat_name = patient_name.strip() if patient_name and patient_name.strip() else "Patient"
    current_date = datetime.now().strftime("%Y-%m-%d")

    audio_content = await audio.read()
    
    transcribed_text = transcribe_audio_hf(audio_content, audio.filename or "audio.wav")

    if not transcribed_text:
        return {
            "status": "error", 
            "message": "Audio transcription failed. Baraye meharbani saaf awaz mein dobara audio upload karein."
        }

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
