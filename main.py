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
    "transcription": "", 
    "summary": "", 
    "doctor": "Dr. Zainab", 
    "patient": "Patient", 
    "date": datetime.now().strftime("%Y-%m-%d")
}

@app.get("/")
def home():
    return {"status": "AI Medical Scribe Backend Active"}

def transcribe_with_assemblyai(audio_bytes: bytes) -> str:
    try:
        headers = {'authorization': '8f27807a0c8b417bbd222e4d03e91d60'}
        upload_response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=audio_bytes)
        
        if upload_response.status_code == 200:
            audio_url = upload_response.json().get('upload_url')
            json_payload = { "audio_url": audio_url, "language_detection": True }
            tx_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_payload, headers=headers)
            
            if tx_response.status_code == 200:
                tx_id = tx_response.json().get('id')
                import time
                for _ in range(15):
                    polling_res = requests.get(f'https://api.assemblyai.com/v2/transcript/{tx_id}', headers=headers)
                    res_json = polling_res.json()
                    if res_json.get('status') == 'completed':
                        return res_json.get('text', '').strip()
                    elif res_json.get('status') == 'error':
                        break
                    time.sleep(1)
    except Exception as e:
        print(f"AssemblyAI Exception: {e}")
    return ""

def transcribe_audio_hf(audio_bytes: bytes, filename: str) -> str:
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    ext = filename.split(".")[-1].lower() if "." in filename else "wav"
    content_type_map = {
        "aac": "audio/aac", "m4a": "audio/m4a", "mp3": "audio/mpeg",
        "ogg": "audio/ogg", "wav": "audio/wav", "webm": "audio/webm"
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": content_type_map.get(ext, "audio/wav")
    }
    try:
        res = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=35)
        if res.status_code == 200:
            result = res.json()
            return result.get("text", "").strip()
    except Exception as e:
        print(f"HF Error: {e}")
    return ""

def process_transcription(audio_bytes: bytes, filename: str) -> str:
    # 1. Primary Transcriber
    text = transcribe_with_assemblyai(audio_bytes)
    if text and len(text.strip()) > 1:
        return text.strip()

    # 2. HF Whisper Fallback
    text_hf = transcribe_audio_hf(audio_bytes, filename)
    if text_hf and len(text_hf.strip()) > 1:
        return text_hf.strip()

    # 3. SpeechRecognition
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language="ur-PK")
    except Exception:
        pass

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""You are an expert AI Medical Scribe assisting a doctor.
Analyze the provided transcript (which may be in Urdu, Roman Urdu, or English).

LAWS:
1. Grounding: Extract ONLY clinical complaints and symptoms explicitly mentioned in the transcript.
2. Translation: Translate spoken Urdu symptoms into formal English.
3. Relevant Treatment: ONLY suggest generic medications or care plans that directly correspond to the explicitly mentioned symptoms in the audio.
4. Unclear Audio Handling: If the transcript is empty or no valid medical complaints are detected, state "No distinct clinical symptoms detected" under Chief Complaint and DO NOT prescribe specific medications.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Extracted spoken symptoms in English]
* **Possible Diagnosis:** [Primary condition matching ONLY the spoken symptoms]

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * [Generic Medication tailored ONLY to detected symptom]: [Dosage & Frequency]
* **Advice/Next Steps:**
    * **Rest:** [Targeted advice]
    * **Hydration:** [Relevant advice]
    * **Monitor Symptoms:** [Warning signs]
    * **Follow-up:** [Timeline]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Audio Transcript: "{transcription_text}"'}
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
            print(f"LLM Error ({model_id}): {err}")
            continue

    # Clean fallback without fake medicines
    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** {transcription_text if transcription_text else "No speech detected in audio"}
* **Possible Diagnosis:** Pending physician clinical evaluation.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
    * Clinical assessment required before issuing prescription.
* **Advice/Next Steps:**
    * **Rest:** General rest advised.
    * **Hydration:** Maintain fluid intake.
    * **Monitor Symptoms:** Monitor condition.
    * **Follow-up:** Consultation with physician."""

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
        filename = audio.filename if audio.filename else "audio.wav"
        
        transcribed_text = process_transcription(audio_content, filename)
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
