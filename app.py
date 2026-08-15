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

BASE_URL = "https://ai-medical-scribe-app.vercel.app"
PROCESS_URL = f"{BASE_URL}/process-audio/"
DOWNLOAD_URL = f"{BASE_URL}/download-pdf/"

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
                
                response = requests.post(PROCESS_URL, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("status") == "success":
                        st.success("Report Generated Successfully!")
                        
                        # Store summary in session state to persist on rerun
                        st.session_state["report_summary"] = res_json.get("summary")
                    else:
                        st.error(res_json.get("message", "Error processing audio."))
                else:
                    st.error(f"HTTP Server Error: Status Code {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Connection Error: {e}")

# Display Report and PDF Download Button if report exists in session state
if "report_summary" in st.session_state:
    st.markdown(st.session_state["report_summary"])
    st.divider()
    
    # Fetch PDF from backend
    try:
        pdf_response = requests.get(DOWNLOAD_URL, timeout=30)
        if pdf_response.status_code == 200:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_response.content,
                file_name="Clinical_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("PDF generated on server but download link couldn't be loaded.")
    except Exception as e:
        st.error(f"Could not load PDF download option: {e}")
