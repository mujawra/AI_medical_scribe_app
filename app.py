import streamlit as st
import requests

st.set_page_config(page_title="Medical Scribe AI", page_icon="🩺", layout="centered")

st.title("AI Medical Scribe App 🩺")
st.write("Professional Consultation & Clinical Documentation System")

st.markdown("---")

# Vercel Live Backend Base URL (Ensure HTTPS)
BACKEND_URL = "https://ai-medical-scribe-app.vercel.app"

doc_input = st.text_input("Enter Doctor's Name:", value="Dr. Zainab", key="doc_name_input")
patient_input = st.text_input("Enter Patient's Name (Type 'Auto-Detect' to let AI find it):", value="Auto-Detect", key="patient_name_input")

uploaded_file = st.file_uploader(
    "Select Audio File (Supported: WAV, MP3, M4A, OGG, AAC)", 
    type=["wav", "mp3", "m4a", "ogg", "aac"],
    key="audio_file_uploader"
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Process & Generate Medical Report 🚀", use_container_width=True, key="process_btn"):
        st.info("AI is parsing clinical data and structure formatting... Please wait.")
        
        st.session_state["generated"] = False
        if "transcription" in st.session_state:
            del st.session_state["transcription"]
        if "summary" in st.session_state:
            del st.session_state["summary"]
            
        files = {"audio": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data_payload = {"doctor_name": doc_input, "patient_name": patient_input}
        
        try:
            # Added trailing slash '/process-audio/' for Vercel route matching
            response = requests.post(f"{BACKEND_URL}/process-audio/", files=files, data=data_payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    st.success("Report successfully generated!")
                    st.session_state["transcription"] = data.get("transcription")
                    st.session_state["summary"] = data.get("summary")
                    st.session_state["generated"] = True
                    st.rerun()
                else:
                    st.error(data.get("message", "Unknown error from backend."))
            else:
                st.error(f"Backend Server Error: {response.status_code}. Response: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: Unable to reach FastAPI backend. Details: {e}")

if st.session_state.get("generated"):
    with st.expander("📝 View Raw Audio Transcription"):
        st.write(st.session_state["transcription"])
    
    st.markdown("---")
    st.markdown(st.session_state["summary"])
    st.markdown("---")
    
    # Safe Lazy-Fetch PDF logic
    pdf_download_url = f"{BACKEND_URL}/download-pdf/"
    try:
        pdf_response = requests.get(pdf_download_url, timeout=30)
        if pdf_response.status_code == 200:
            st.download_button(
                label="📥 Download Official Clinical Report PDF",
                data=pdf_response.content,
                file_name="Official_Clinical_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_btn"
            )
        else:
            st.error("Backend PDF service is initializing. Please click process again.")
    except Exception:
        st.warning("PDF Data rendering endpoint offline.")
