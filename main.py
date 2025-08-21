import os
import re
import shutil
import tempfile
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Transformers (local inference)
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
)
import torch

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Env -------------------
load_dotenv()


# ------------------- FastAPI -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Globals -------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Summarization model (BART)
logger.info("Loading summarization model: facebook/bart-large-cnn ...")
summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# QA model (Flan-T5)
logger.info("Loading QA model: google/flan-t5-base ...")
qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
qa_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
summ_model.to(device)
qa_model.to(device)

# Vector store + raw text
vectorstore = None
full_corpus_text = ""

# ------------------- Helpers -------------------

def clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")  
    t = t.replace("\u200b", "")  
    t = t.replace("\r", " ")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def bart_summarize_block(text: str, max_len: int = 380, min_len: int = 120) -> str:
    inputs = summ_tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)
    summary_ids = summ_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=6,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

def summarize_long_document(full_text: str) -> str:
    """Chunk text, summarize each, then merge into one final summary"""
    text = clean_text(full_text)
    if not text:
        return "The document appears to be empty."

    return bart_summarize_block(text, max_len=420, min_len=160)

def format_bullet_summary(summary: str) -> list:
    """Convert summary text into a clean list of bullet points"""
    sentences = re.split(r'(?<=[.!?]) +', summary)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return bullets

def flan_answer_with_context(context: str, question: str) -> str:
    prompt = (
        "You are a helpful assistant. Use ONLY the given context to answer the question. "
        "If the answer cannot be found in the context, say \"I don't know.\" "
        f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = qa_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    outputs = qa_model.generate(
        **inputs,
        max_length=256,
        min_length=32,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
    )
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

# ------------------- Upload Endpoint -------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, full_corpus_text
    try:
        logger.info(f"📂 Received file: {file.filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            ext = file.filename.lower()
            if ext.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif ext.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif ext.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            docs = loader.load()
            if not docs:
                raise HTTPException(status_code=400, detail="No readable content found in the file")

            full_text_parts = [clean_text(d.page_content) for d in docs if d.page_content and d.page_content.strip()]
            full_corpus_text = " ".join(full_text_parts).strip()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
            split_docs = splitter.split_documents(docs)
            logger.info(f"✅ Split into {len(split_docs)} chunks.")
            vectorstore = FAISS.from_documents(split_docs, embeddings)

        return {"message": f"{file.filename} uploaded and indexed successfully."}

    except Exception as e:
        logger.exception("Error during upload")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Query Endpoint -------------------
@app.post("/query/")
async def query_bot(question: str = Form(...)):
    global vectorstore, full_corpus_text

    if vectorstore is None or not full_corpus_text:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a file first.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    q = question.strip()
    logger.info(f"🔍 Question: {q}")

    try:
        if "summarize" in q.lower() or "summary" in q.lower():
            logger.info("🧾 Summarizing whole document...")
            raw_summary = summarize_long_document(full_corpus_text)
            bullets = format_bullet_summary(raw_summary)
            return {"answer": bullets}

        logger.info("📚 Retrieving context for QA...")
        docs = vectorstore.similarity_search(q, k=5)
        if not docs:
            return {"answer": ["I don't know."]}

        context = " ".join(clean_text(d.page_content) for d in docs)
        if len(context) > 12000:
            context = context[:12000]

        answer = flan_answer_with_context(context, q)
        return {"answer": [answer]}

    except Exception as e:
        logger.exception("Error during query")
        raise HTTPException(status_code=500, detail=str(e))
