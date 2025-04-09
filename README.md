# Medical AI Chatbot — Streamlit + Groq + LangChain + FAISS

This project implements a Retrieval-Augmented Generation (RAG)-based medical chatbot with a focus on accuracy, contextual relevance, and real-time response capabilities. Built using Streamlit, LangChain, Groq LLMs, and FAISS, the chatbot provides users with fact-based medical answers by retrieving data from trusted sources such as medical PDFs and authoritative web pages.

## Key Features

- Retrieval-augmented architecture using LangChain and FAISS
- Real-time semantic search over documents (PDF + Web)
- Conversational memory for contextual chat flow
- Uses Groq-hosted LLM (deepseek-r1-distill-llama-70b) for generation
- Streamlit-based UI for an interactive user experience
- Modular and scalable design for future enhancements

## System Architecture
User (Streamlit Chat Input) │ ▼ LangChain RAG Pipeline ├── Retriever (FAISS Vector Store) ├── Prompt Template (includes context + chat history) ├── Chat Memory (InMemoryChatMessageHistory) └── Groq LLM (deepseek-r1-distill-llama-70b) ▼ Answer Returned to User


## Data Sources

- PDF Document:
  - Local medical reference file (e.g., azencyclopedia.pdf)
- Web Pages:
  - NHS Inform: Illnesses and Conditions A–Z
  - Drugs.com: Drug Information Database

These documents are parsed and semantically chunked before embedding and indexing into FAISS.

## Technology Stack

| Component         | Description                                       |
|------------------|---------------------------------------------------|
| UI                | Streamlit                                        |
| LLM               | Groq (deepseek-r1-distill-llama-70b)             |
| Framework         | LangChain                                        |
| Embeddings        | HuggingFace MiniLM (`all-MiniLM-L6-v2`)          |
| Vector Store      | FAISS                                            |
| Web Scraping      | BeautifulSoup via WebBaseLoader                  |
| PDF Parsing       | PyPDFLoader                                      |
| Memory Handling   | LangChain `InMemoryChatMessageHistory`           |
| Environment Config| Python-dotenv (`.env` file)                      |

## Directory Structure
medical-chatbot/ ├── app.py # Main Streamlit app ├── .env # Contains GROQ_API_KEY ├── azencyclopedia.pdf # PDF medical reference ├── requirements.txt # Python dependencies └── README.md # Documentation


## Example Interaction

**User:** What are the common symptoms of anemia?  
**MedBot:** Symptoms include fatigue, weakness, pale skin, shortness of breath, and cold hands or feet.

**User:** What medicine is prescribed?  
**MedBot:** Iron supplements such as ferrous sulfate or ferrous gluconate are commonly used. Please consult a healthcare provider before use.

## Planned Enhancements

- Voice input support with speech-to-text transcription
- OCR-based analysis of uploaded medical report images
- Hospital and clinic locator based on user geolocation
- Medical knowledge graph integration for improved diagnosis
- Performance monitoring dashboard with user and model analytics
- Feedback loop to improve chatbot accuracy over time
- Integration of real-time medical APIs and databases

## License

This project is licensed under the MIT License.



