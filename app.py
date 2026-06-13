import os
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
os.environ.setdefault("USER_AGENT", "telecom-rag-chatbot")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

try:
    from langchain_classic.chains import RetrievalQA
except (ImportError, ModuleNotFoundError):
    from langchain.chains import RetrievalQA

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="📡 Telecom RAG Chatbot", layout="wide")

# -----------------------------
# SIMPLE CLEAN CSS
# -----------------------------
st.markdown("""
<style>
h1 {
    text-align: center;
    font-size: 40px;
    background: linear-gradient(90deg, #00c3ff, #b56eff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📡 Telecom RAG Chatbot")

    uploaded_files = st.file_uploader(
        "📂 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    url_input = st.text_area(
        "🌐 Add Web Links",
        placeholder="https://example.com"
    )

    if st.button("🔄 Refresh Knowledge Base"):
        st.cache_resource.clear()
        st.success("Refreshed!")

# -----------------------------
# HERO IMAGE
# -----------------------------
st.markdown("""
<div style="
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/3/3f/Base_station_antennas.jpg');
    background-size: cover;
    padding: 70px;
    border-radius: 12px;
"></div>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE + TEXT
# -----------------------------
st.markdown("<h1>📡 Telecom RAG Chatbot</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;'>
    <div style='font-size:20px; color:#444444; font-weight:500;'>
        Upload documents and start asking questions
    </div>
    <br>
    <div style='font-size:16px; color:#666666;'>
        Upload any number of <b>PDFs</b> or add <b>web links</b> to build your knowledge base.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Ask your telecom question</h3>", unsafe_allow_html=True)

# -----------------------------
# DOCUMENT LOADING
# -----------------------------
def load_documents(files, urls):
    docs = []

    # PDFs
    if files:
        for file in files:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file.name)
            docs.extend(loader.load())

    # URLs
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except:
            st.warning(f"Failed: {url}")

    return docs

# -----------------------------
# VECTOR STORE (OPTIMIZED)
# -----------------------------
@st.cache_resource
def create_vectorstore(urls, file_names):
    docs = load_documents(uploaded_files, urls)

    # 🔥 Reduced chunk size (cost optimization)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

# -----------------------------
# PROMPT (COST + QUALITY)
# -----------------------------
PROMPT = PromptTemplate(
    template="""
Answer the question using ONLY the provided context.
Keep the answer concise and clear.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# -----------------------------
# QA CHAIN (OPTIMIZED)
# -----------------------------
@st.cache_resource
def get_qa_chain(urls, file_names):
    vectorstore = create_vectorstore(urls, file_names)

    # 🔥 Optimized LLM
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0,
        max_completion_tokens=300
    )

    # 🔥 Reduced retrieval (major cost saver)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
query = st.chat_input("Type your telecom question...")

# -----------------------------
# RUN QUERY
# -----------------------------
if query:
    urls = tuple([u.strip() for u in url_input.split("\n") if u.strip()])
    file_names = tuple([f.name for f in uploaded_files]) if uploaded_files else ()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("📡 Analyzing..."):

            qa_chain = get_qa_chain(urls, file_names)
            result = qa_chain(query)

            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(answer)

            # Sources (trimmed display)
            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources):
                    st.write(f"Source {i+1}")
                    st.write(doc.page_content[:200] + "...")
                    st.write("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
