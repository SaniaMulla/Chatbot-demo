import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="📚 AI Chatbot - Document Q&A", layout="centered")
st.title("📚 AI Chatbot - Document Q&A")

# File upload
uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/upload/", files=files)
        if response.status_code == 200:
            st.success(response.json().get("message"))
        else:
            st.error(f"{response.status_code} - {response.json().get('detail')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

# Ask question (triggers on Enter)
question = st.text_input("Ask a question about the uploaded document:")

if question: 
    try:
        response = requests.post(f"{BACKEND_URL}/query/", data={"question": question})
        if response.status_code == 200:
            answer = response.json()["answer"]

            st.write("### Answer:")
            if isinstance(answer, list):  
                for point in answer:
                    st.write(point)
            else:
                st.write(answer)
        else:
            st.error("Error: " + response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
