import streamlit as st
import requests
import speech_recognition as sr
from datetime import datetime
from pydub import AudioSegment
from fpdf import FPDF
import io
import tempfile
import os

st.set_page_config(page_title="Medical Scribe AI", page_icon="🩺", layout="centered")

st.title("AI Medical Scribe App 🩺")
st.write("Professional Consultation & Clinical Documentation System")

st.markdown("---")

# Read token from Streamlit Secrets or hardcoded fallback
HF_TOKEN = st.secrets.get("HF_TOKEN", "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")

doc_input = st.text_input("Enter Doctor's Name:", value="Dr. Zainab")
patient_input = st.text_input("Enter Patient's Name (Type 'Auto-Detect' to let AI find it):", value="Auto-Detect")

uploaded_file = st.file_uploader("Select Audio File (Supported: WAV, MP3, M4A, OGG, AAC)", type=["wav", "mp3", "m4a", "ogg", "aac"])

# Audio conversion helper
def convert_audio_to_wav(file_bytes, input_ext):
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=input_ext)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception:
        return io.BytesIO(file_bytes)

# Direct Call to Hugging Face API
def generate_medical_report_hf(transcription_text, doctor_name, patient_name):
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

STRICT INSTRUCTIONS FOR CONTENT:
1. Names: Transliterate Doctor Name and Patient Name into clean English script.
2. Zero Unrelated Symptoms: Include ONLY complaints mentioned in transcript. Do NOT add unmentioned conditions.

Format strictly as:

### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** (Summary based strictly on transcript)

### 📝 Recommended Plan

* **Advice/Next Steps:** (Based strictly on transcript)"""
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
                if "choices" in result and len(result["choices"]) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
        except Exception:
            continue

    return f"Audio Transcript: {transcription_text}\n\nNote: AI evaluation failed."

def clean_txt_for_pdf(text: str) -> str:
    return text.replace("**", "").replace("###", "").replace("📋", "").replace("🩺", "").replace("📝", "").encode('latin-1', 'ignore').decode('latin-1')

def generate_robust_pdf(summary_text, transcription_text, doc_name, pat_name, report_date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_fill_color(26, 54, 93)
    pdf.rect(0, 0, 210, 32, 'F')
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "CLINICAL CONVERSATION REPORT", ln=True, align="C")
    pdf.ln(12)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(45, 55, 72)
    
    safe_summary = clean_txt_for_pdf(summary_text)
    pdf.multi_cell(0, 7, safe_summary.strip())
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Raw Audio Transcript:", ln=True)
    
    pdf.set_font("Helvetica", "I", 9)
    safe_transcript = clean_txt_for_pdf(transcription_text)
    pdf.multi_cell(0, 5, f'"{safe_transcript}"')
    
    return pdf.output(dest='S').encode('latin-1')

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Process & Generate Medical Report 🚀", use_container_width=True):
        st.info("AI is parsing clinical data and structure formatting... Please wait.")
        
        # Reset previous session state to clear older errors/reports
        st.session_state["generated"] = False
        if "transcription" in st.session_state:
            del st.session_state["transcription"]
        if "summary" in st.session_state:
            del st.session_state["summary"]
            
        file_bytes = uploaded_file.read()
        file_ext = uploaded_file.name.split(".")[-1].lower()
        wav_io = convert_audio_to_wav(file_bytes, file_ext)
        
        text_result = ""
        try:
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(wav_io.getvalue())
                temp_wav_path = temp_wav.name

            with sr.AudioFile(temp_wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)
                try:
                    text_result = recognizer.recognize_google(audio_data, language="ur-PK")
                except Exception:
                    text_result = recognizer.recognize_google(audio_data, language="en-US")
                    
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception:
            text_result = "Audio received, but voice could not be converted to text clearly."

        doc_name = doc_input.strip() if doc_input and doc_input.strip() else "Dr. Zainab"
        pat_name = patient_input.strip() if patient_input and patient_input.strip() else "Auto-Detect"
        current_date = datetime.now().strftime("%Y-%m-%d")

        summary_text = generate_medical_report_hf(text_result, doc_name, pat_name)

        st.session_state["transcription"] = text_result
        st.session_state["summary"] = summary_text
        st.session_state["doctor"] = doc_name
        st.session_state["patient"] = pat_name
        st.session_state["date"] = current_date
        st.session_state["generated"] = True
        
        st.rerun()

# Render report if successfully generated
if st.session_state.get("generated"):
    with st.expander("📝 View Raw Audio Transcription"):
        st.write(st.session_state["transcription"])
    
    st.markdown("---")
    st.markdown(st.session_state["summary"])
    st.markdown("---")
    
    try:
        pdf_bytes = generate_robust_pdf(
            st.session_state["summary"],
            st.session_state["transcription"],
            st.session_state["doctor"],
            st.session_state["patient"],
            st.session_state["date"]
        )
        st.download_button(
            label="📥 Download Official Clinical Report PDF",
            data=pdf_bytes,
            file_name="Official_Clinical_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
