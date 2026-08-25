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
    # 1. First Try Google Speech Recognition (Urdu & English) - Highly Accurate for Speech
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            # Urdu Try
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            if text and len(text.strip()) > 1:
                return text.strip()
    except Exception as e:
        print(f"Urdu SR Error: {e}")

    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            # English Try
            text = recognizer.recognize_google(audio_data, language="en-US")
            if text and len(text.strip()) > 1:
                return text.strip()
    except Exception as e:
        print(f"English SR Error: {e}")

    # 2. Backup HF Whisper API
    text_hf = transcribe_audio_hf(audio_bytes)
    if text_hf and len(text_hf.strip()) > 1:
        return text_hf.strip()

    return ""

def generate_medical_report(transcription_text, doctor_name, patient_name):
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # If audio transcription was completely empty, don't call AI LLM
    if not transcription_text or transcription_text.strip() == "":
        return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** Audio sound was unclear or empty.
* **Possible Diagnosis:** Please record the audio clearly again.

### 📝 Recommended Plan

* **Medication (Rough AI Suggestion — NOT a prescription):** Not applicable — no audio detected to process.
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor before giving anything to the patient.
* **Advice/Next Steps:**
    * **Rest:** Re-record speaking clearly into the microphone.
    * **Hydration:** N/A
    * **Monitor Symptoms:** N/A
    * **Follow-up:** Re-submit clear audio recording."""

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI medical scribe assisting a doctor by turning a spoken consultation into structured clinical notes with a rough, non-final medication idea.

RULES (STRICT):
1. ONLY extract symptoms, complaints, history, or observations explicitly spoken in the audio transcript. Do not invent or assume anything not said.
2. Translate Urdu/Roman Urdu spoken text into professional English.
3. You may suggest ONE common, generic, low-risk over-the-counter medicine class typically associated with the stated symptom (e.g. a general antipyretic/analgesic for fever/pain, an antacid for indigestion, ORS for dehydration) as a ROUGH IDEA ONLY — not a specific brand, not a dosage or frequency, and not for anything beyond simple, everyday symptoms.
4. If the transcript mentions anything serious or ambiguous (chest pain, breathing difficulty, severe/persistent symptoms, pregnancy, children, high fever, symptoms lasting many days, or anything you are not confident about), do NOT suggest any medicine — write "Doctor must evaluate before any medication" instead.
5. Never give a dosage, frequency, or duration under any circumstances — this always requires the doctor's judgment based on age, weight, and history not available from voice alone.
6. If the patient mentions a medicine they already took, record it as history in Chief Complaint only — do not repeat or endorse it in the plan.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** [Translate spoken symptoms/history to English accurately, including any medicines the patient says they already took, reported as history only]
* **Possible Diagnosis:** [Primary differential(s) suggested by the spoken complaint, phrased as "to be confirmed by physician"]

### 📝 Recommended Plan

* **Medication (Rough AI Suggestion — NOT a prescription):** [One generic medicine class only if symptom is simple/common, per Rule 3–4, with no dosage. If not applicable, write "Doctor must evaluate before any medication."]
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor before giving anything to the patient.
* **Advice/Next Steps:**
    * **Rest:** [General, non-drug guidance for this issue]
    * **Hydration:** [Relevant general fluid/dietary guidance]
    * **Monitor Symptoms:** [Key warning signs for this issue]
    * **Follow-up:** [Timeline for re-consultation with the doctor]"""
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

* **Chief Complaint:** {transcription_text}
* **Possible Diagnosis:** Evaluation required based on transcript.

### 📝 Recommended Plan

* **Medication (Rough AI Suggestion — NOT a prescription):** Doctor must evaluate before any medication.
* **⚠️ Disclaimer:** This is an AI-generated rough idea only. It is not a prescription and must be reviewed and confirmed by the doctor before giving anything to the patient.
* **Advice/Next Steps:**
    * **Rest:** General rest advised.
    * **Hydration:** Maintain hydration.
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
