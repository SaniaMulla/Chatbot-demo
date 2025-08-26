import os
import re
import io
import shutil
import tempfile
import logging
import streamlit as st
import numpy as np
import torch
import faiss

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Streamlit page ----------
st.set_page_config(page_title="📚 AI Chatbot - Document Q&A", layout="centered")
st.title("📚 AI Chatbot - Document Q&A")

# ---------- Globals in session_state ----------
if "index" not in st.session_state:
    st.session_state.index = None        # FAISS index
if "chunks" not in st.session_state:
    st.session_state.chunks = []         # parallel list of chunk texts
if "full_text" not in st.session_state:
    st.session_state.full_text = ""      # entire document text

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Small utilities ----------
def clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")      # soft hyphen
    t = t.replace("\u200b", "")      # zero-width space
    t = t.replace("\r", " ")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 120):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------- Cached model loaders ----------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_summarizer():
    from transformers import BartTokenizer, BartForConditionalGeneration
    tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    mdl = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    return tok, mdl

@st.cache_resource(show_spinner=False)
def get_qa_model():
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
    return tok, mdl

# ---------- File readers (no LangChain) ----------
def read_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    txt_parts = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            txt_parts.append(t)
    return "\n".join(txt_parts)

def read_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)

def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---------- Embedding / FAISS ----------
def build_faiss_index(chunks, embed_model):
    vecs = embed_model.encode(chunks, show_progress_bar=True)
    vecs = np.array(vecs).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    return index

def search_chunks(question: str, k: int = 5):
    if st.session_state.index is None or not st.session_state.chunks:
        return []
    embed = get_embedding_model()
    q_vec = embed.encode([question]).astype("float32")
    D, I = st.session_state.index.search(q_vec, k)
    return [st.session_state.chunks[i] for i in I[0] if 0 <= i < len(st.session_state.chunks)]

# ---------- Summarization / QA ----------
def summarize_text(full_text: str) -> str:
    if not full_text.strip():
        return "The document appears to be empty."
    tok, mdl = get_summarizer()
    text = clean_text(full_text)
    inputs = tok.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    ids = mdl.generate(
        inputs,
        max_length=420, min_length=160,
        length_penalty=2.0, num_beams=6, early_stopping=True, no_repeat_ngram_size=3
    )
    return tok.decode(ids[0], skip_special_tokens=True).strip()

def answer_with_context(context: str, question: str) -> str:
    qa_tok, qa_mdl = get_qa_model()
    prompt = (
        "You are a helpful assistant. Use ONLY the given context to answer the question. "
        "If the answer cannot be found in the context, say \"I don't know.\" "
        f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = qa_tok(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    ids = qa_mdl.generate(
        **inputs, max_length=256, min_length=24,
        num_beams=5, early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.2
    )
    return qa_tok.decode(ids[0], skip_special_tokens=True).strip()

def bullets(text: str):
    sents = re.split(r'(?<=[.!?]) +', text)
    return [f"• {s.strip()}" for s in sents if s.strip()]

# ---------- UI: Upload ----------
uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded is not None:
    with st.spinner("Loading models and processing document…"):
        try:
            # Save to temp and read
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, uploaded.name)
                with open(path, "wb") as f:
                    shutil.copyfileobj(uploaded, f)

                name = uploaded.name.lower()
                if name.endswith(".pdf"):
                    raw_text = read_pdf(path)
                elif name.endswith(".docx"):
                    raw_text = read_docx(path)
                elif name.endswith(".txt"):
                    raw_text = read_txt(path)
                else:
                    st.error("Unsupported file type")
                    st.stop()

            cleaned = clean_text(raw_text)
            st.session_state.full_text = cleaned

            # Chunk + embed + faiss
            chunks = chunk_text(cleaned, chunk_size=500, overlap=120)
            st.session_state.chunks = chunks

            embed = get_embedding_model()
            st.session_state.index = build_faiss_index(chunks, embed)

            st.success(f"{uploaded.name} uploaded and indexed successfully.")
        except Exception as e:
            st.error(f"Error during upload: {e}")

# ---------- UI: Question ----------
question = st.text_input("Ask a question about the uploaded document (or type 'summarize'):")

if question:
    if st.session_state.index is None or not st.session_state.full_text:
        st.error("Please upload a document first.")
    else:
        try:
            if "summarize" in question.lower():
                st.info("🧾 Summarizing the whole document…")
                summary = summarize_text(st.session_state.full_text)
                for b in bullets(summary):
                    st.write(b)
            else:
                st.info("🔎 Retrieving relevant passages…")
                top_chunks = search_chunks(question, k=5)
                context = " ".join(top_chunks)
                if len(context) > 12000:
                    context = context[:12000]
                answer = answer_with_context(context, question)
                st.write("### Answer:")
                st.write(answer)
        except Exception as e:
            st.error(f"Error during query: {e}")
