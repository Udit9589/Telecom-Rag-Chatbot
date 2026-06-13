# 📡 Telecom RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **OpenAI**, and **Streamlit**, designed to answer questions about telecom documentation.

Upload your own PDFs or add web links — the chatbot builds a vector knowledge base on the fly and answers questions grounded in your documents.

## Architecture

![Telecom RAG Chatbot](sample_output/Telecom_RAG_Chatbot.png)
---

## 🖼️ Preview

![Telecom RAG Chatbot](sample_output/Telecom_RAG_Chatbot.png)

---

## ✨ Features

- **PDF ingestion** — upload one or many PDFs directly in the UI
- **Web scraping** — add URLs as additional knowledge sources
- **FAISS vector store** — fast in-memory semantic search
- **OpenAI embeddings + GPT** — context-grounded answers
- **Source attribution** — every answer shows the retrieved source chunks
- **Session-based chat memory** — conversation history within a session
- **Cost-optimised** — small chunk size (500 tokens), top-3 retrieval, token cap on responses

---

## 🏗️ Architecture

```
User Input (PDF / URL / Question)
        │
        ▼
Document Loader  ──►  Text Splitter (chunk_size=500, overlap=100)
        │
        ▼
OpenAI Embeddings  ──►  FAISS Vector Store
        │
        ▼
Retriever (top-k=3)  ──►  PromptTemplate  ──►  ChatOpenAI (GPT)
        │
        ▼
Answer + Source Documents  ──►  Streamlit Chat UI
```

---

## 🗂️ Project Structure

```
telecom-rag-chatbot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
│
├── data/                   # Sample telecom PDFs (add your own)
│   ├── Assisted Time Holdover.pdf
│   ├── Basic AAS for FDD.pdf
│   └── Basic AAS for TDD.pdf
│
├── docs/                   # Architecture & design notes
│   └── architecture.md
│
├── sample_output/          # Screenshots / demo output
│   └── Telecom_RAG_Chatbot.png
│
└── snippets/               # Code snippets (add your own)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/telecom-rag-chatbot.git
cd telecom-rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS / Linux
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
# Open .env and set your OpenAI API key
```

```env
OPENAI_API_KEY=sk-...
```

> ⚠️ **Never commit `.env` to Git.** It is already in `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Usage

1. **Upload PDFs** via the sidebar file uploader.
2. **Add web URLs** (one per line) in the sidebar text area.
3. **Ask questions** in the chat input at the bottom.
4. Expand **📚 Sources** under any answer to see the retrieved context chunks.
5. Click **🔄 Refresh Knowledge Base** to reload documents after changes.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Token size of each document chunk |
| `chunk_overlap` | 100 | Overlap between adjacent chunks |
| `k` (retrieval) | 3 | Number of chunks retrieved per query |
| `max_completion_tokens` | 300 | Max tokens in LLM response |
| `temperature` | 0 | Deterministic output |
| `model` | `gpt-4o-mini` | OpenAI chat model |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `langchain` | RAG orchestration |
| `langchain-community` | PDF & web loaders, FAISS |
| `langchain-openai` | OpenAI embeddings & chat |
| `faiss-cpu` | Local vector similarity search |
| `pypdf` | PDF text extraction |
| `openai` | OpenAI API client |
| `python-dotenv` | Load `.env` variables |
| `beautifulsoup4` + `requests` | Web page scraping |

---

## 🔐 Security Notes

- Store your OpenAI API key in `.env` only — never hardcode it.
- `.env` is excluded from version control by `.gitignore`.
- Uploaded PDFs are saved temporarily in the working directory at runtime; they are not persisted or tracked.

---

## 🗺️ Roadmap

- [ ] Persistent vector store (save/load FAISS index)
- [ ] Multi-user session isolation
- [ ] Support for DOCX and TXT ingestion
- [ ] Streaming LLM responses
- [ ] Docker deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
