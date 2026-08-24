from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import requests
import io
import base64
import tempfile
from datetime import datetime
from fpdf import FPDF
from pydub import AudioSegment

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
    "transcription": "", 
    "summary": "", 
    "doctor": "Dr. Zainab", 
    "patient": "Patient", 
    "date": datetime.now().strftime("%Y-%m-%d")
}

def convert_to_wav(audio_bytes: bytes, filename: str) -> bytes:
    """Converts mobile audio formats (.aac, .m4a, etc.) to PCM WAV in memory"""
    ext = os.path.splitext(filename)[1].lower().replace('.', '')
    if not ext:
        ext = "aac"
    
    # Return as-is if already clean wav
    if ext == "wav":
        return audio_bytes

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
        out_io = io.BytesIO()
        audio.export(out_io, format="wav")
        return out_io.getvalue()
    except Exception as e:
        print(f"Audio conversion fallback (pydub): {e}")
        return audio_bytes

def transcribe_audio_robust(audio_bytes: bytes, filename: str) -> str:
    """Robust transcription that handles mobile .aac/.m4a formats"""
    # 1. Convert any mobile audio to standard WAV first
    clean_wav_bytes = convert_to_wav(audio_bytes, filename)
    
    # 2. Try Whisper Large v3
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/wav"
    }
    
    try:
        res = requests.post(API_URL, headers=headers, data=clean_wav_bytes, timeout=35)
        if res.status_code == 200:
            result = res.json()
            if isinstance(result, dict) and "text" in result:
                text = result.get("text", "").strip()
                if text:
                    return text
    except Exception as e:
        print(f"Whisper API error: {e}")

    # 3. Fallback AssemblyAI Direct API
    try:
        headers_aai = {'authorization': '8f27807a0c8b417bbd222e4d03e91d60'}
        upload_res = requests.post('https://api.assemblyai.com/v2/upload', headers=headers_aai, data=clean_wav_bytes)
        if upload_res.status_code == 200:
            audio_url = upload_res.json().get('upload_url')
            tx_res = requests.post(
                'https://api.assemblyai.com/v2/transcript', 
                json={"audio_url": audio_url, "language_detection": True}, 
                headers=headers_aai
            )
            if tx_res.status_code == 200:
                tx_id = tx_res.json().get('id')
                import time
                for _ in range(12):
                    poll = requests.get(f'https://api.assemblyai.com/v2/transcript/{tx_id}', headers=headers_aai)
                    p_json = poll.json()
                    if p_json.get('status') == 'completed':
                        return p_json.get('text', '').strip()
                    elif p_json.get('status') == 'error':
                        break
                    time.sleep(1)
    except Exception as e:
        print(f"AssemblyAI fallback error: {e}")

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""You are an expert AI Clinical Scribe.
Analyze the audio transcript provided (spoken in Urdu, Roman Urdu, or English).

INSTRUCTIONS:
1. Extract clinical symptoms from transcript.
2. If audio is unclear/empty, provide a default general assessment without refusal messages.
3. Detect condition and specify generic medicines with dosages.
4. Provide instructions (Hadaiyat).
5. Output in clear English.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Spoken symptoms translated to English]
* **Possible Diagnosis:** [Condition detected]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Generic Medication & Dosage]: [Usage instructions]
* **Advice/Next Steps (Hadaiyat):**
    * **Rest:** [Targeted advice]
    * **Hydration & Diet:** [Guidance]
    * **Monitor Symptoms:** [Warning signs]
    * **Follow-up:** [Timeline]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Audio Transcript: "{transcription_text if transcription_text else "Patient presented with general symptoms."}"'}
    ]

    models = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]

    for model_id in models:
        try:
            res = requests.post(
                ROUTER_URL, 
                headers=headers, 
                json={"model": model_id, "messages": messages, "temperature": 0.2, "max_tokens": 800}, 
                timeout=25
            )
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
        except Exception:
            continue

    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** {transcription_text if transcription_text else "General clinical evaluation."}
* **Possible Diagnosis:** Symptomatic assessment required.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * Consultation required before prescribing medication.
* **Advice/Next Steps (Hadaiyat):**
    * **Rest:** Rest advised.
    * **Hydration & Diet:** Increase fluid intake.
    * **Monitor Symptoms:** Track symptoms closely.
    * **Follow-up:** Consult physician."""

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

@app.get("/")
def home():
    return {"status": "AI Medical Scribe API Active"}

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
        filename = audio.filename if audio.filename else "audio.aac"
        
        # 1. Transcribe with mobile audio conversion
        transcribed_text = transcribe_audio_robust(audio_content, filename)
        
        # 2. Generate Report
        summary_text = generate_medical_report(transcribed_text, doc_name, pat_name)

        # 3. PDF Base64
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
