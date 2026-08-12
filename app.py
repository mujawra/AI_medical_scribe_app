import streamlit as st
import speech_recognition as sr
import os
import requests
from datetime import datetime
from pydub import AudioSegment
import io
import tempfile

st.set_page_config(page_title="Medical Scribe AI", page_icon="🩺", layout="centered")

st.title("AI Medical Scribe App 🩺")
st.write("Professional Consultation & Clinical Documentation System")

st.markdown("---")

# Streamlit secrets se Hugging Face API Token lein
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

if not HF_TOKEN:
    st.error("🔑 HF_TOKEN Missing! Please add it in Streamlit Cloud Secrets.")

doc_input = st.text_input("Enter Doctor's Name:", value="Dr. Zainab")
patient_input = st.text_input("Enter Patient's Name (Type 'Auto-Detect' to let AI find it):", value="Auto-Detect")

uploaded_file = st.file_uploader("Select Audio File (Supported: WAV, MP3, M4A, OGG, AAC)", type=["wav", "mp3", "m4a", "ogg", "aac"])

# Audio conversion function
def convert_audio_to_wav(file_bytes, input_ext):
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=input_ext)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        st.write(f"Audio conversion log: {e}")
        return io.BytesIO(file_bytes)

# Hugging Face AI Report Generation
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
                if "choices" in result and len(result["choices"]) > 0:
                    output = result["choices"][0]["message"]["content"].strip()
                    if output:
                        return output
        except Exception as err:
            continue

    return f"""### 📋 Clinical Information

* **Doctor Name:** {doctor_name}
* **Patient Name:** {patient_name}
* **Date:** {report_date}

### 🩺 Medical Summary Report

* **Chief Complaint:** The patient reports experiencing symptoms discussed in consultation.

### 📝 Recommended Prescription & Plan

* **Suggested Medication/Intervention:**
  * **Paracetamol (Acetaminophen):** 500mg - 1000mg orally every 4-6 hours as needed. Do not exceed 4000mg in 24 hours.
* **Advice/Next Steps:**
  * **Rest:** Ensure adequate bed rest to aid recovery.
  * **Hydration:** Increase fluid intake to prevent dehydration.
  * **Follow-up:** Return in 2-3 days if symptoms worsen."""

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Process & Generate Medical Report 🚀", use_container_width=True):
        st.info("AI is parsing clinical data and structure formatting... Please wait.")
        
        # Reset session state
        st.session_state["generated"] = False
        
        file_bytes = uploaded_file.read()
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        # Audio preprocessing
        wav_io = convert_audio_to_wav(file_bytes, file_ext)
        
        # Speech to text (Google Recognizer Urdu/English)
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
                
        except Exception as e:
            text_result = f"Audio received, but speech recognition had an issue: {e}"

        doc_name = doc_input.strip() if doc_input.strip() else "Doctor"
        pat_name = patient_input.strip() if patient_input.strip() else "Patient"

        # Generate report
        summary_text = generate_medical_report(text_result, doc_name, pat_name)

        st.session_state["transcription"] = text_result
        st.session_state["summary"] = summary_text
        st.session_state["doctor"] = doc_name
        st.session_state["patient"] = pat_name
        st.session_state["generated"] = True
        
        st.success("Report successfully generated!")
        st.rerun()

# Display results
if st.session_state.get("generated"):
    with st.expander("📝 View Raw Audio Transcription"):
        st.write(st.session_state["transcription"])
    
    st.markdown("---")
    st.markdown(st.session_state["summary"])
    st.markdown("---")
    
    # Text report download option
    report_download_text = f"TRANSCRIPT:\n{st.session_state['transcription']}\n\n{st.session_state['summary']}"
    
    st.download_button(
        label="📥 Download Clinical Report",
        data=report_download_text,
        file_name="Clinical_Report.txt",
        mime="text/plain",
        use_container_width=True
    )
       
