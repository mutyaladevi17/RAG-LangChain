# Insurellm RAG: Baseline LangChain Implementation

![RAG Pipeline](https://img.shields.io/badge/Architecture-RAG-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Python](https://img.shields.io/badge/Language-Python_3.10+-yellow)

## Overview
This repository contains the baseline **Retrieval-Augmented Generation (RAG)** system for an insurance company named "Insurellm". It demonstrates a highly structured standard RAG pipeline built with LangChain, ChromaDB, and OpenAI. A fundamental engineering focus of this project is evaluating the impact of different document chunking strategies (Small, Large, and Hybrid) on overall retrieval recall and generation accuracy.

---

## Features
* **LangChain Integration:** Leverages the LangChain framework for seamlessly orchestrating the retrieval strings and conversational generation pipeline.
* **Vector Storage:** Uses persistent ChromaDB collections to store and retrieve dense document embeddings locally.
* **Chunking Strategy Comparison:** Built-in infrastructure to toggle between three different vector store configurations seamlessly:
  * Small DB (k=10 chunks retrieved)
  * Large DB (k=5 chunks retrieved)
  * Hybrid DB (k=7 chunks retrieved)
* **Gradio Chat UI:** A sleek, dual-pane interactive web interface to converse with the language model and inspect the exact retrieved knowledge chunks.
* **Automated Evaluation Dashboard:** A standalone Gradio analytics application for evaluating the LangChain RAG pipeline across 150 benchmark tests using LLM-as-a-judge metrics (MRR, nDCG, Keyword Coverage, Accuracy, Completeness, Relevance).

---

## Architecture & Tech Stack
* **LLM Engine:** ChatOpenAI (`gpt-4.1-nano`)
* **Embeddings:** OpenAI Embeddings (`text-embedding-3-large`)
* **Vector Database:** ChromaDB
* **UI Framework:** Gradio
* **Eval Framework:** Custom metrics logic + LLM-as-a-judge 

---

## Pipeline Step-by-Step

This repository implements a robust baseline approach to retrieval augmented generation:

1. **Document Loading & Preprocessing (`ingest.py`):** 
   Documents derived from the Insurellm knowledge base are loaded and natively split into chunks based on predefined size limits (e.g., small, large, or hybrid). This prepares the text for downstream processing.
2. **Embedding Generation:** 
   The raw textual chunks are passed through a deep learning embedding model (`text-embedding-3-large`) mapped into a high-dimensional vector space, and deposited into one of three distinct local ChromaDB collections.
3. **Conversational Vector Retrieval (`answer.py`):** 
   Upon receiving a user query, the interactive conversational history is compacted. The text is embedded, and an approximate nearest neighbor (ANN) search retrieves the top `k` semantically similar chunks via mathematical distance (Cosine Similarity/L2). 
4. **Context Construction:**
   The `k` retrieved chunks are aggressively concatenated into a unified system prompt context block, grounding the model in the enterprise's private data.
5. **Generation Output:**
   A LangChain ChatModel (`ChatOpenAI`) interprets the augmented prompt to formulate a heavily constrained, fact-based response back to the user interface.

---

## Prerequisites

Ensure you have the following installed on your local machine:
* Python 3.10+
* `uv` or `pip` package manager

---

## Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   cd RAG-LangChain
   ```

2. **Install the dependencies:**
   ```bash
   uv pip install -r requirements.txt
   # OR
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Prepare the Vector Database:**
   If you have new documents, run the ingestion script to populate the Chroma datasets:
   ```bash
   python implementation/ingest.py
   ```

---

## Usage

### 1. Chat Application
To start the interactive chat agent:
```bash
python app_gradio.py
```
This will launch a Gradio interface locally (usually at `http://127.0.0.1:7860`).

### 2. Evaluation Dashboard
To run the evaluation suite comparing Small, Large, and Hybrid vector databases:
```bash
python evaluator_comparison.py
```

---

## Evaluation Results

The system was evaluated against 150 benchmark tests. The performance comparison between chunking strategies is as follows:

| Metric | Small DB (k=10) | Large DB (k=5) | Hybrid DB (k=7) |
| :--- | :---: | :---: | :---: |
| **Retrieval MRR** | 0.7834 | 0.8541 | 0.8367 |
| **Retrieval nDCG** | 0.7977 | 0.8636 | 0.8504 |
| **Keyword Coverage** | 87.8% | **92.7%** | 91.7% |
| **Answer Accuracy** | 4.11/5 | **4.45/5** | 4.27/5 |
| **Answer Completeness** | 3.96/5 | **4.23/5** | 4.11/5 |
| **Answer Relevance** | 4.67/5 | **4.81/5** | 4.69/5 |

*Insight: The **Large DB** strategy yielded the most factually accurate and complete answers within a basic LangChain architecture, primarily by minimizing context fragmentation.*

---

## Project Structure

```text
RAG-LangChain/                
├── app_gradio.py              # Gradio Chat application
├── evaluator_comparison.py    # Generates the multi-DB evaluation dashboard
├── implementation/
│   ├── answer.py              # LangChain RAG pipeline logic
│   └── ingest.py              # Script to ingest docs into Chroma DBs
├── evaluation/
│   ├── eval.py                # Scoring metrics (MRR, nDCG, LLM Judge)
│   └── test.py                # Benchmark loading logic
├── vector_db_small/           # Persistent Chroma directory (Small Chunks)
├── vector_db_large/           # Persistent Chroma directory (Large Chunks)
└── vector_db_hybrid/          # Persistent Chroma directory (Hybrid Chunks)
```

---

## Limitations & Future Work

* **Context Fragmentation:** Base vector search can miss scattered information. The Large DB mitigate this slightly, but query expansion could yield better results.
* **Lexical Gap:** Dense embeddings occasionally miss exact keyword matching if the user vocabulary is drastically different from the document corpus.
* **Future Work:** This repository serves as a baseline. See the accompanying `RAG-advancedRetrevial` project for complex implementations of document summarization, query rewriting and LLM-based reranking intended to solve these hard constraints natively.
