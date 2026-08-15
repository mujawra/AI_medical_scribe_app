import streamlit as st
import requests

st.set_page_config(page_title="AI Medical Scribe", page_icon="🩺", layout="wide")

st.title("🩺 AI Medical Scribe")
st.write("Upload audio to generate structured clinical reports.")

# 🔗 BACKEND URL (Without trailing slash)
BACKEND_URL = "https://ai-medical-scribe-app.vercel.app"

col1, col2 = st.columns(2)
with col1:
    doctor_name = st.text_input("Enter Doctor's Name:", "Dr. Zainab")
with col2:
    patient_name = st.text_input("Enter Patient's Name:", "Auto-Detect")

audio_file = st.file_uploader("Select Audio File (Supported: WAV, MP3, M4A, OGG)", type=["wav", "mp3", "m4a", "ogg"])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Process & Generate Medical Report 🚀", type="primary"):
        with st.spinner("AI is parsing clinical data and structure formatting... Please wait."):
            try:
                # Prepare payload
                files = {"audio": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                data_payload = {
                    "doctor_name": doctor_name, 
                    "patient_name": patient_name
                }
                
                # Request to FastAPI Backend
                response = requests.post(
                    f"{BACKEND_URL}/process-audio/", 
                    files=files, 
                    data=data_payload, 
                    timeout=90
                )
                
                if response.status_code == 200:
                    res_data = response.json()
                    
                    if res_data.get("status") == "success":
                        st.success("Report Generated Successfully!")
                        
                        st.subheader("🗣️ Detected Audio Transcript")
                        st.info(res_data.get("transcription", "No text transcribed."))
                        
                        st.subheader("📄 Generated Clinical Report")
                        st.markdown(res_data.get("summary", ""))
                        
                        pdf_download_url = f"{BACKEND_URL}/download-pdf"
                        st.markdown(f"### [📥 Download Official PDF Report]({pdf_download_url})")
                    else:
                        st.error(res_data.get("message", "Audio processing failed."))
                else:
                    st.error(f"Backend Returned HTTP Error {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                st.error("Error: Connection to FastAPI backend refused. Check if Vercel server URL is accessible.")
