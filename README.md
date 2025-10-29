# 🎬 Mini RAG System — Movie Plots

A lightweight **Retrieval-Augmented Generation (RAG)** system that answers questions about movie plots from a small subset of the **Wikipedia Movie Plots** dataset.

This project demonstrates the full **data → embeddings → retrieval → LLM → structured output** pipeline using a minimal, memory-only setup.

---

## 🚀 Overview

The system:
1. Loads and preprocesses a **500-row subset** of movie plots.
2. Splits long plots into ~300-word chunks for efficient retrieval.
3. Converts text chunks into **vector embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores and retrieves vectors using an **in-memory Chroma** vector store.
5. Queries a **Groq LLM** to generate natural-language answers based on retrieved context.
6. Outputs a stru## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/mini-rag-movie-plots.git
cd mini-rag-movie-plotsctured JSON containing:
   - `answer`: a short, natural-language answer  
   - `contexts`: retrieved plot snippets  
   - `reasoning`: a brief explanation of how the answer was formed
```

---
### 2️⃣ Create a conda environment

---

### 3️⃣ Install dependencies
``` bash
pip install -r requirements.txt
```
---
### 4️⃣ Set up your Groq API key

Create a .env file in the project root:
``` bash
GROQ_API_KEY=your_api_key_here
```

---

### ▶️ Running the Program

Run the RAG system from the command line:
``` bash
python main.py
```

You’ll see:
``` bash
Type your question below (or type 'exit' to quit)
```


