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
def transcribe_audio_hf(audio_bytes: bytes) -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("CRITICAL ERROR: HF_TOKEN missing!")
        return ""

    # Hugging Face Whisper Endpoints
    models = [
        "https://api-inference.huggingface.co/models/openai/whisper-large-v3",
        "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
        "https://api-inference.huggingface.co/models/openai/whisper-small"
    ]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream"  # Binary raw stream format for HF
    }

    for api_url in models:
        for attempt in range(2):  # Auto retry if model is loading
            try:
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    data=audio_bytes, 
                    params={"wait_for_model": "true"}, 
                    timeout=45
                )
                print(f"HF Model: {api_url.split('/')[-1]} | Status: {response.status_code}")
                
                if response.status_code == 200:
                    res = response.json()
                    text = ""
                    if isinstance(res, dict) and "text" in res:
                        text = res["text"].strip()
                    elif isinstance(res, list) and len(res) > 0 and "text" in res[0]:
                        text = res[0]["text"].strip()
                    
                    if text and len(text) > 2:
                        return text
                elif response.status_code == 503:
                    # Model loading, wait and retry once
                    import time
                    time.sleep(5)
                    continue
                else:
                    print(f"HF Error Output: {response.text}")
            except Exception as e:
                print(f"Exception on {api_url}: {e}")
                break

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
Analyze the audio transcript and generate a structured clinical report.

STRICT INSTRUCTIONS:
1. DO NOT write or prescribe any medications or drug names.
2. Extract symptoms, illness, and patient complaints strictly from the audio transcript.
3. Transliterate Roman Urdu/Hindi spoken words into English accurately.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Detailed clinical summary extracted strictly from transcript]

### 📝 Recommended Plan & Advice

* **General Care:** [Self-care guidance and lifestyle advice based on symptoms]
* **Follow-up:** [Timeline for checkup]"""
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

    return f"### 📋 Clinical Information\n\n* **Doctor Name:** {doctor_name}\n* **Patient Name:** {patient_name}\n* **Date:** {report_date}\n\n### 🩺 Medical Summary Report\n\n* **Chief Complaint:** {transcription_text}\n\n### 📝 Recommended Plan & Advice\n\n* **General Care:** Rest and hydration recommended.\n* **Follow-up:** Please follow up with physician if symptoms persist."

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
    
    transcribed_text = transcribe_audio_hf(audio_content)

    if not transcribed_text:
        return {
            "status": "error", 
            "message": "Hugging Face model audio read nahi kar saka. Please clear voice recorder audio upload karein."
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
