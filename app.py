import gradio as gr
import google.generativeai as genai
import faiss
import numpy as np
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Gemini for embeddings only (lightweight, won't hit quota)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Groq for generation (fast + free)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    chunks = []
    size, overlap = 1000, 200
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def embed_text(text, task="retrieval_document"):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task
    )
    return result["embedding"]

print(" Loading paper...")
chunks = load_pdf("paper.pdf")
print(f" {len(chunks)} chunks created. Building FAISS index...")
embeddings = np.array([embed_text(c) for c in chunks], dtype="float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f" Ready!")

def answer(question, history):
    start = time.time()
    query_vec = np.array([embed_text(question, task="retrieval_query")], dtype="float32")
    _, indices = index.search(query_vec, k=4)
    context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are an expert assistant on the research paper 'Attention Is All You Need'.
Answer clearly and in detail using ONLY the context below.
If the answer is not in the context, say 'This isn't covered in the paper.'

Context:
{context}"""},
            {"role": "user", "content": question}
        ],
        max_tokens=1024
    )

    elapsed = round(time.time() - start, 2)
    answer_text = response.choices[0].message.content
    return f"{answer_text}\n\n* {elapsed}s | {len(indices[0])} chunks from FAISS*"

demo = gr.ChatInterface(
    fn=answer,
    title=" AI Research Paper Chatbot",
    description="""**Ask anything about 'Attention Is All You Need'** — the 2017 paper that invented the Transformer powering ChatGPT, Gemini, and all modern AI.

Built with: **RAG + FAISS + Groq (Llama 3.3 70B) + Gradio**""",
    examples=[
        "What is the main contribution of this paper?",
        "Explain the attention mechanism in simple terms",
        "What are multi-head attention and why is it used?",
        "What BLEU scores did the model achieve?",
        "How is positional encoding implemented?",
        "What are the encoder and decoder components?"
    ],
)

if __name__ == "__main__":
    demo.launch()