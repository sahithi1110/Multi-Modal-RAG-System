# **Multi-Modal RAG System**

A **production-style Multi-Modal Retrieval-Augmented Generation (RAG) system** that supports both **text and image understanding** using modern AI pipelines.

This project demonstrates how to build an end-to-end system that combines **hybrid retrieval, query optimization, re-ranking, and guardrails** to generate accurate and grounded responses.

---

## **Key Features**

* Supports **text + image retrieval**
* Uses **hybrid search (FAISS + BM25)** for higher accuracy
* Implements **query rewriting** for better search relevance
* Uses **Cross-Encoder re-ranking** to improve result quality
* Applies **guardrails** to reduce hallucinations
* Provides **real-time FastAPI endpoints**
* Works with or without **OpenAI API**

---

## **Tech Stack**

* **FastAPI** – API layer
* **FAISS** – vector similarity search
* **BM25** – keyword-based retrieval
* **Sentence Transformers** – text embeddings
* **CLIP** – image embeddings
* **Cross-Encoder** – re-ranking
* **OpenAI API (optional)** – answer generation

---

## **Project Structure**

```
multimodal-rag-system/
│
├── app/              # API layer
├── src/              # core pipeline logic
├── scripts/          # ingestion and data generation
├── tests/            # test files
├── data/             # raw data
├── artifacts/        # indexes and metadata
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## **Setup (Windows)**

```powershell
py -3.14 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## **Environment Setup**

```powershell
Copy-Item .env.example .env
```

(Optional) Add your API key:

```
OPENAI_API_KEY=your_key_here
```

---

## **Run the Project**

### Step 1: Create Sample Data

```powershell
python -m scripts.create_sample_data
```

### Step 2: Build Indexes

```powershell
python -m scripts.ingest_data
```

### Step 3: Start Server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## **Access the Application**

* [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## **API Usage**

### **POST /ask**

```json
{
  "question": "How does the system reduce hallucinations?",
  "top_k": 5,
  "include_images": true
}
```

---

## **How It Works**

1. User query is **rewritten** for better retrieval
2. System performs **hybrid search (FAISS + BM25)**
3. Results are **re-ranked using Cross-Encoder**
4. Guardrails filter unsafe or irrelevant responses
5. Final answer is generated using:

   * **LLM (if API key is provided)**
   * or **context-based fallback logic**

---

## **Evaluation**

```powershell
python -m scripts.run_eval
```

---

## **Notes**

* First run downloads models from **Hugging Face**
* Works even without OpenAI (fallback answers)
* Image retrieval uses **CLIP similarity**
* Designed to simulate **real-world RAG pipelines**

---

## **What This Project Demonstrates**

* End-to-end **RAG system design**
* Handling **multi-modal data (text + images)**
* Combining **dense + sparse retrieval**
* Reducing hallucination using **re-ranking + guardrails**
* Building **scalable inference APIs**

---

## **Author**

**Sahithi**

---

