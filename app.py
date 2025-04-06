import streamlit as st
import os
from dotenv import load_dotenv
import bs4
import re
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ API Key is missing! Please set it in your .env file.")
    st.stop()

# ✅ Sidebar
st.sidebar.title("Medical Chatbot - Knowledge Base")
show_thoughts = st.sidebar.checkbox("🧠 Show internal thoughts", value=False)

# ✅ Set Absolute Path for PDF
PDF_PATH = r"C:\\Users\\balak\\gen_ai\\rag\\azencyclopedia.pdf"
if not os.path.exists(PDF_PATH):
    st.error(f"❌ PDF file '{PDF_PATH}' not found. Please check the file path.")
    st.stop()

@st.cache_data()
def load_data():
    st.sidebar.text("Fetching data...")
    pdf_loader = PyPDFLoader(PDF_PATH)
    pdf_docs = pdf_loader.load()

    web_loader = WebBaseLoader(
        web_paths=[
            "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/",
            "https://www.drugs.com/drug_information.html"
        ],
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
    )
    web_docs = web_loader.load()

    return pdf_docs + web_docs

documents = load_data()
st.sidebar.success(f"✅ Loaded {len(documents)} documents from PDF & Web.")

# ✅ Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# ✅ Cache FAISS Database
@st.cache_resource()
def get_faiss_db():
    st.sidebar.text("Generating embeddings...")
    return FAISS.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

db = get_faiss_db()
st.sidebar.success("✅ Vector database ready!")

# ✅ Retrieval setup
retriever = db.as_retriever()

# ✅ Smart Prompt with strict context usage
debug_prompt = ChatPromptTemplate.from_template("""
You are MedBot, a medically-focused AI assistant.

Use ONLY the information provided in <context> to answer the user's medical questions.
If you cannot find the answer in the context, politely say so.

<context>
{context}
</context>

Chat History:
{chat_history}

User: {question}
MedBot:
""")

# ✅ LLM and Chain Setup
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
document_chain = create_stuff_documents_chain(llm, debug_prompt)
rag_chain = RunnableMap(
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
) | document_chain

# ✅ Conversational memory setup
if "chat_ui_history" not in st.session_state:
    st.session_state.chat_ui_history = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = InMemoryChatMessageHistory()

# ✅ Conversational wrapper
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history=lambda session_id: st.session_state.chat_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ✅ Helper to strip <think> block if disabled
def extract_visible_answer(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ✅ Streamlit UI
st.title("🩺 Medical AI Chatbot")
query = st.chat_input("Ask a medical question or say hi...")

# ✅ Print old messages before new query
for msg in st.session_state.chat_ui_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# ✅ Handle new message
if query:
    st.session_state.chat_ui_history.append({"user": query, "bot": ""})  # add placeholder
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting MedBot..."):
            response = conversational_rag_chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "med-session"}},
            )
            raw_answer = response["answer"] if isinstance(response, dict) else response
            answer = raw_answer if show_thoughts else extract_visible_answer(raw_answer)
            st.markdown(answer)

    st.session_state.chat_ui_history[-1]["bot"] = answer  # update last entry with actual bot response
