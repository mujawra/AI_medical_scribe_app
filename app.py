import streamlit as st
import requests
import base64

st.set_page_config(page_title="AI Medical Scribe", page_icon="🩺", layout="wide")

st.title("🩺 AI Medical Scribe & Diagnostic System")
st.write("Upload or record patient audio (Urdu / Roman Urdu / English) to automatically extract disease, prescription & instructions.")

BACKEND_URL = "https://your-backend-url.onrender.com"  # Apne backend ka URL yahan likhein

col1, col2 = st.columns(2)
with col1:
    doctor_name = st.text_input("Doctor Name", "Dr. Zainab")
with col2:
    patient_name = st.text_input("Patient Name", "Patient")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "aac", "ogg", "webm"])

if st.button("Process & Generate Medical Report 🚀"):
    if uploaded_file is not None:
        with st.spinner("Analyzing Audio & Generating Diagnosis..."):
            try:
                files = {"audio": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"doctor_name": doctor_name, "patient_name": patient_name}
                
                response = requests.post(f"{BACKEND_URL}/process-audio", files=files, data=data)
                
                if response.status_code == 200:
                    res_json = response.json()
                    st.success("Report Generated Successfully!")
                    
                    # Show Transcription
                    with st.expander("🗣️ Spoken Text (Transcription)"):
                        st.write(res_json.get("transcription", "No text detected"))
                    
                    # Display Markdown Summary
                    st.markdown(res_json.get("summary", ""))
                    
                    # Download PDF Button
                    pdf_b64 = res_json.get("pdf_base64")
                    if pdf_b64:
                        pdf_bytes = base64.b64decode(pdf_b64)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"Medical_Report_{patient_name}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.error(f"Error from server: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
    else:
        st.warning("Please upload an audio file first!")
