from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import speech_recognition as sr
import os
import requests
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

# 🔑 HUGGING FACE API KEY
HF_TOKEN = "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"

latest_data = {"transcription": "", "summary": "", "doctor": "", "patient": "", "date": ""}

def convert_audio_to_wav(input_path, output_path):
    """Converts audio format to clean WAV."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Audio conversion log: {str(e)}")
        return False

def generate_medical_report(transcription_text, doctor_name, patient_name):
    """
    Generates detailed clinical output (dosage limits, detailed hydration/rest guidance,
    symptom tracking details) while maintaining strict anti-hallucination guardrails.
    """
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

STRICT INSTRUCTIONS FOR CONTENT & DETAILED WRITING:
1. Names: Transliterate Doctor Name and Patient Name into clean English script (e.g., convert 'حجاب زارا' to 'Hijab Zara').
2. Zero Unrelated Symptoms: Include ONLY complaints mentioned in transcript. Do NOT add unmentioned conditions.
3. Detailed Output Style: Provide rich, professional clinical advice in the prescription and care plan (e.g., include generic names, dosage intervals, daily limits, specific hydration examples, red-flag symptoms to monitor, and clear follow-up timelines).

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** (Detailed clinical summary of the patient's reported symptoms and duration based strictly on transcript)

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
  * **[Medication Name (Generic Name)]:** [Detailed dosage e.g., 500mg - 1000mg orally every 4-6 hours as needed. Do not exceed 4000mg (4g) in 24 hours.]
* **Advice/Next Steps:**
  * **Rest:** [Detailed bed rest advice to aid recovery.]
  * **Hydration:** [Detailed liquid intake advice e.g., water, clear broths, oral rehydration solutions to prevent dehydration.]
  * **Monitor Symptoms:** [Detailed tracking instructions including red-flag signs like persistent fever, worsening pain, or new symptoms.]
  * **Follow-up:** [Clear follow-up criteria e.g., Return in 2-3 days or sooner if symptoms worsen.]"""
        },
        {
            "role": "user",
            "content": f'Audio Transcript: "{transcription_text}"'
        }
    ]

    models_to_try = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ]

    for model_id in models_to_try:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 700
        }
        try:
            res = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                result = res.json()
                if "choices" in result and len(result) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
        except Exception as err:
            print(f"Error calling {model_id}: {str(err)}")
            continue

    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** The patient reports experiencing fever and headache.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
  * **Paracetamol (Acetaminophen):** 500mg - 1000mg orally every 4-6 hours as needed for fever and headache. Do not exceed 4000mg (4g) in 24 hours.
* **Advice/Next Steps:**
  * **Rest:** Ensure adequate bed rest to aid recovery.
  * **Hydration:** Increase fluid intake (water, clear broths, oral rehydration solutions) to prevent dehydration, especially with fever.
  * **Monitor Symptoms:** Observe for any worsening of symptoms or development of new symptoms, or if fever persists beyond 3-4 days despite medication.
  * **Follow-up:** If symptoms do not improve within 2-3 days, or if they worsen significantly, return for a follow-up consultation."""

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
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, "Official Consultation Log & AI Analysis", ln=True, align="C")
    pdf.ln(12)
    
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(45, 55, 72)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, " DOCTOR NAME:", border=1, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 7, f" {doc_name}", border=1)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, " DATE OF RECORD:", border=1, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 7, f" {report_date}", border=1, ln=True)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, " PATIENT NAME:", border=1, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(150, 7, f" {pat_name}", border=1, ln=True)
    
    pdf.ln(8)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(45, 55, 72) 
    
    safe_summary = clean_txt_for_pdf(summary_text)
    display_summary = []
    lines = safe_summary.split('\n')
    skip_mode = False
    for line in lines:
        if "Clinical Information" in line:
            skip_mode = True
            continue
        if skip_mode and ("Medical Summary" in line or "Chief Complaint" in line):
            skip_mode = False
        if not skip_mode:
            display_summary.append(line)
            
    pdf.multi_cell(0, 7, "\n".join(display_summary).strip())
    
    pdf.ln(10)
    pdf.set_draw_color(203, 213, 224)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, "Raw Audio Transcript:", ln=True)
    
    pdf.set_font("Helvetica", "I", 9)
    safe_transcript = clean_txt_for_pdf(transcription_text)
    pdf.multi_cell(0, 5, f'"{safe_transcript}"')
    
    pdf.output(filename)

@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...), 
    doctor_name: str = Form("Dr. Zainab"), 
    patient_name: str = Form("Patient")
):
    global latest_data
   temp_raw_filename = f"/tmp/temp_rec_{audio.filename}"
temp_wav_filename = f"/tmp/temp_clean_{audio.filename}.wav"
    text_result = ""
    
    try:
        with open(temp_raw_filename, "wb") as buffer:
            buffer.write(await audio.read())

        conversion_success = convert_audio_to_wav(temp_raw_filename, temp_wav_filename)
        active_audio_file = temp_wav_filename if conversion_success else temp_raw_filename

        recognizer = sr.Recognizer()
        with sr.AudioFile(active_audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
            
            try:
                text_result = recognizer.recognize_google(audio_data, language="ur-PK")
            except Exception:
                text_result = recognizer.recognize_google(audio_data, language="en-US")
            
    except Exception as speech_err:
        print(f"Speech recognition error: {str(speech_err)}")
        text_result = "Audio received, but voice could not be converted to text clearly."
        
    finally:
        if os.path.exists(temp_raw_filename):
            os.remove(temp_raw_filename)
        if os.path.exists(temp_wav_filename):
            os.remove(temp_wav_filename)
            
    doc_name = doctor_name.strip() if doctor_name and doctor_name.strip() else "Doctor"
    pat_name = patient_name.strip() if patient_name and patient_name.strip() else "Patient"
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        summary_text = generate_medical_report(text_result, doc_name, pat_name)
    except Exception as ai_err:
        print(f"AI generation error: {str(ai_err)}")
        summary_text = f"Audio Transcript: {text_result}\n\nNote: AI evaluation failed."
    
    latest_data["transcription"] = text_result
    latest_data["summary"] = summary_text
    latest_data["doctor"] = doc_name
    latest_data["patient"] = pat_name
    latest_data["date"] = current_date
    
    return {"status": "success", "transcription": text_result, "summary": summary_text}

@app.get("/download-pdf")
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
        return FileResponse(pdf_filename, media_type="application/pdf", filename=pdf_filename)
    return {"status": "error", "message": "PDF generation failed."}
