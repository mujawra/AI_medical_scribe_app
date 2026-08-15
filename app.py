import streamlit as st
import requests

st.set_page_config(page_title="AI Medical Scribe", page_icon="🩺", layout="centered")

st.title("🩺 AI Medical Scribe")

col1, col2 = st.columns(2)
with col1:
    doctor_name = st.text_input("Enter Doctor's Name:", value="Dr. Zainab")
with col2:
    patient_name = st.text_input("Enter Patient's Name:", value="Auto-Detect")

uploaded_file = st.file_uploader(
    "Select Audio File (Supported: WAV, MP3, M4A, OGG, AAC, FLAC, WMA)", 
    type=["wav", "mp3", "m4a", "ogg", "aac", "flac", "wma"]
)

BACKEND_URL = "https://ai-medical-scribe-app.vercel.app/process-audio/"

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Process & Generate Medical Report 🚀"):
        with st.spinner("Processing audio and generating clinical report..."):
            try:
                files = {
                    "audio": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                data = {
                    "doctor_name": doctor_name,
                    "patient_name": patient_name
                }
                
                response = requests.post(BACKEND_URL, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("status") == "success":
                        st.success("Report Generated Successfully!")
                        st.markdown(res_json.get("summary"))
                    else:
                        st.error(res_json.get("message", "Error processing audio."))
                else:
                    st.error(f"HTTP Server Error: Status Code {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Connection Error: {e}")
