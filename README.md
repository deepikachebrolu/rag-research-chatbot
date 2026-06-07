#  RAG Research Paper Chatbot
An AI-powered chatbot that answers questions over research papers using Retrieval-Augmented Generation (RAG).

##  Live Demo
**Hugging Face Spaces:** https://huggingface.co/spaces/dheep8/rag-research-chatbot

##  How It Works
1. PDF is loaded and split into 1000-token chunks with 200-token overlap
2. Google Gemini embeds each chunk into vectors
3. FAISS IndexFlatL2 indexes all vectors for similarity search
4. User asks a question → Gemini embeds the query
5. FAISS retrieves top 4 most relevant chunks
6. Groq (Llama 3.3 70B) generates an answer from the context
7. Response returned in <3 seconds

##  Stack
| Layer | Tech |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | Google Gemini Embedding |
| Vector Store | FAISS (IndexFlatL2) |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

##  Current Paper
**"Attention Is All You Need"** — Vaswani et al. (2017)  
The paper that invented the Transformer architecture powering ChatGPT, Gemini, and all modern AI.

##  Run Locally
```bash
git clone https://github.com/deepikachebrolu/rag-research-chatbot
cd rag-research-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Add .env with GEMINI_API_KEY and GROQ_API_KEY
# Add paper.pdf to root folder
python app.py
```

##  Example Questions
- What is the main contribution of this paper?
- Explain the attention mechanism in simple terms
- What BLEU scores did the model achieve?
- How does multi-head attention work?
- What is positional encoding?
