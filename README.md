# ESTIN RAG System 🎓

A **Retrieval-Augmented Generation (RAG)** system designed to answer questions about the internal regulations of ESTIN (École Supérieure en Sciences et Technologies de l'Informatique et du Numérique).

## 🎯 What is RAG?

RAG (Retrieval-Augmented Generation) is an AI architecture that enhances Large Language Models (LLMs) by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the LLM's context with these documents
3. **Generating** accurate, grounded responses

This prevents "hallucinations" and ensures answers are based on actual ESTIN regulations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Embedding                               │
│              (Convert question to vector)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Store Search                           │
│            (Find similar document chunks)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Generation                                │
│         (Generate answer using retrieved context)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Answer                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
RI_ESTIN_RAG/
├── src/
│   ├── config/          # Configuration and settings
│   ├── data_processing/ # Document loading and chunking
│   ├── embeddings/      # Text embedding utilities
│   ├── vectorstore/     # Vector database operations
│   ├── rag/             # RAG chain and agent logic
│   └── api/             # FastAPI endpoints
├── data/
│   └── documents/       # ESTIN regulation documents (PDF, etc.)
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for experimentation
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
└── docker-compose.yml   # Multi-container setup
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RI_ESTIN_RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   uvicorn src.api.main:app --reload
   ```

## 🛠️ Technologies

- **LangChain v1.x** - LLM application framework
- **FastAPI** - Modern web framework
- **ChromaDB/FAISS** - Vector database
- **OpenAI** - Embeddings and LLM
- **Docker** - Containerization

## 📚 Learning Resources

This project serves as a learning resource for AI Engineering. Each module contains educational comments explaining:
- Why certain design decisions were made
- How components work together
- Best practices in production AI systems

## 📄 License

MIT License - See LICENSE file for details.

