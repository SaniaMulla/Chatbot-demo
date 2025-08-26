import os
import re
import shutil
import tempfile
import logging
import streamlit as st
import torch

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Globals -------------------
vectorstore = None
full_corpus_text = ""
embeddings = None
summ_model = None
summ_tokenizer = None
qa_model = None
qa_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- Helpers -------------------

def clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")  
    t = t.replace("\u200b", "")  
    t = t.replace("\r", " ")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_models():
    global embeddings, summ_model, summ_tokenizer, qa_model, qa_tokenizer, device
    if embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if summ_model is None:
        from transformers import BartTokenizer, BartForConditionalGeneration
        summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    if qa_model is None:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        qa_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

def bart_summarize_block(text: str, max_len: int = 380, min_len: int = 120) -> str:
    inputs = summ_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = summ_model.generate(
        inputs, max_length=max_len, min_length=min_len,
        length_penalty=2.0, num_beams=6, early_stopping=True, no_repeat_ngram_size=3
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

def summarize_long_document(full_text: str) -> str:
    text = clean_text(full_text)
    if not text:
        return "The document appears to be empty."
    return bart_summarize_block(text, max_len=420, min_len=160)

def format_bullet_summary(summary: str) -> list:
    sentences = re.split(r'(?<=[.!?]) +', summary)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return bullets

def flan_answer_with_context(context: str, question: str) -> str:
    prompt = (
        "You are a helpful assistant. Use ONLY the given context to answer the question. "
        "If the answer cannot be found in the context, say \"I don't know.\" "
        f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = qa_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = qa_model.generate(
        **inputs, max_length=256, min_length=32,
        num_beams=5, early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.2
    )
    return qa_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="📚 AI Chatbot - Document Q&A", layout="centered")
st.title("📚 AI Chatbot - Document Q&A")

# File upload
uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    st.info("Loading models and processing document… please wait ⏳")
    load_models()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            ext = uploaded_file.name.lower()
            if ext.endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
            elif ext.endswith(".docx"):
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            elif ext.endswith(".txt"):
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path)
            else:
                st.error("Unsupported file type")
                st.stop()

            docs = loader.load()
            if not docs:
                st.error("No readable content found in the file")
                st.stop()

            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS

            full_text_parts = [clean_text(d.page_content) for d in docs if d.page_content and d.page_content.strip()]
            full_corpus_text = " ".join(full_text_parts).strip()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
            split_docs = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(split_docs, embeddings)

        st.success(f"{uploaded_file.name} uploaded and indexed successfully.")
    except Exception as e:
        st.error(f"Error during upload: {e}")

# Ask question
question = st.text_input("Ask a question about the uploaded document:")

if question:
    if vectorstore is None or not full_corpus_text:
        st.error("Please upload a document first.")
    else:
        try:
            if "summarize" in question.lower() or "summary" in question.lower():
                st.info("🧾 Summarizing whole document…")
                raw_summary = summarize_long_document(full_corpus_text)
                bullets = format_bullet_summary(raw_summary)
                st.write("### Summary:")
                for b in bullets:
                    st.write(b)
            else:
                st.info("📚 Retrieving context for QA…")
                docs = vectorstore.similarity_search(question, k=5)
                if not docs:
                    st.write("I don't know.")
                else:
                    context = " ".join(clean_text(d.page_content) for d in docs)
                    if len(context) > 12000:
                        context = context[:12000]

                    answer = flan_answer_with_context(context, question)
                    st.write("### Answer:")
                    st.write(answer)
        except Exception as e:
            st.error(f"Error during query: {e}")
