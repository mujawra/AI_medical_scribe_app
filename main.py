from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")

@app.get("/")
def read_root():
    return {"status": "FastAPI Backend is Live on Vercel!"}

@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...),
    doctor_name: str = Form("Dr. Zainab"),
    patient_name: str = Form("Auto-Detect")
):
    try:
        if not HF_TOKEN:
            return {"status": "error", "message": "HF_TOKEN missing on Vercel environment variables."}
            
        audio_bytes = await audio.read()
        
        # Updated Hugging Face Router API Endpoint
        API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        hf_response = requests.post(API_URL, headers=headers, data=audio_bytes)
        
        if hf_response.status_code == 200:
            result = hf_response.json()
            transcription = result.get("text", "Audio processed successfully.")
        else:
            return {"status": "error", "message": f"Hugging Face API Error ({hf_response.status_code}): {hf_response.text}"}

        summary = f"### 📋 Clinical Information\n* **Doctor Name:** {doctor_name}\n* **Patient Name:** {patient_name}\n* **Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n### 🩺 Medical Summary & Transcription\n{transcription}"

        return {
            "status": "success",
            "transcription": transcription,
            "summary": summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/download-pdf")
def download_pdf():
    return {"status": "PDF service ready"}
