import os
from pathlib import Path
from typing import Any, List

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"


def load_streamlit_secrets() -> None:
    if hasattr(st, "secrets") and isinstance(st.secrets, dict):
        for key in [
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
            "LLM_PROVIDER",
            "LLM_MODEL",
            "EMBEDDING_MODEL",
        ]:
            value = st.secrets.get(key)
            if value:
                os.environ.setdefault(key, value)


def get_llm_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").strip().lower()


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")


def get_embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT)

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions related to the company's HR policies "
    "and documents."
)


def find_pdf_directory() -> Path:
    if any(PDF_DIR.glob("*.pdf")):
        return PDF_DIR

    raise FileNotFoundError(
        f"No PDF files found in {PDF_DIR}. "
        "Place the HR corpus PDF files in the same folder as app.py."
    )


def load_documents(pdf_dir: Path):
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
    except ImportError as error:
        raise ImportError(
            "Unable to import PyPDFDirectoryLoader from langchain_community.document_loaders. "
            "Please install langchain-community in requirements.txt: langchain-community==0.4.2"
        ) from error

    loader = PyPDFDirectoryLoader(str(pdf_dir))
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No documents could be loaded from {pdf_dir}.")
    return documents


def build_embeddings():
    import torch
    try:
        from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    except ImportError as error:
        raise ImportError(
            "Unable to import SentenceTransformerEmbeddings from langchain_community.embeddings. "
            "Please install langchain-community and sentence-transformers in requirements.txt."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformerEmbeddings(
        model_name=get_embedding_model(),
        model_kwargs={"device": device},
    )


def build_retriever(documents: List[Any]):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as error:
        raise ImportError(
            "Unable to import RecursiveCharacterTextSplitter from langchain_text_splitters. "
            "Please add langchain-text-splitters to requirements.txt if missing."
        ) from error

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as error:
        raise ImportError(
            "Unable to import FAISS from langchain_community.vectorstores. "
            "Please install langchain-community in requirements.txt: langchain-community==0.4.2"
        ) from error

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=768,
        chunk_overlap=128,
        separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = build_embeddings()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
    )


def initialize_llm():
    load_streamlit_secrets()
    provider = get_llm_provider()
    model_name = get_llm_model()

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for GROQ provider.")
        os.environ["GROQ_API_KEY"] = api_key

        return ChatGroq(
            model=model_name,
            temperature=0.1,
            max_tokens=512,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        os.environ["OPENAI_API_KEY"] = api_key

        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=512,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider.")
        os.environ["GOOGLE_API_KEY"] = api_key

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            max_output_tokens=512,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Use groq, openai, or gemini."
    )


@st.cache_resource
def load_rag_components():
    pdf_dir = find_pdf_directory()
    documents = load_documents(pdf_dir)
    retriever = build_retriever(documents)
    llm = initialize_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are an HR Policy Assistant.

Use ONLY the provided context to answer the question.

Rules:
1. Answer only from the retrieved context.
2. If the answer is not present in the context, respond:
   "I could not find that information in the HR policies."
3. Do not make assumptions or hallucinate information.
4. Be concise and professional.
5. When possible, mention the policy source.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_documents(docs: List[Any]) -> str:
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    oos_prompt = ChatPromptTemplate.from_template(
        """
You are a classifier.

Determine whether the question is related to employee HR policies.

HR policy topics include:
- Employee handbook
- Leave policy
- Work from home
- Compensation and benefits
- Performance reviews
- Code of conduct
- IT and data security
- Travel and expenses
- Onboarding
- Separation
- Sexual harassment policy
- Company policies

Respond with ONLY:

YES
or

NO

Question:
{question}
"""
    )

    oos_chain = (
        oos_prompt
        | llm
        | StrOutputParser()
    )

    def ask(question: str) -> str:
        classification = oos_chain.invoke({"question": question}).strip().upper()
        if classification != "YES":
            return REFUSAL_MESSAGE
        return rag_chain.invoke(question)

    return {
        "ask": ask,
        "document_count": len(documents),
        "pdf_dir": str(pdf_dir),
        "model": get_llm_model(),
        "provider": get_llm_provider(),
        "embedding_model": get_embedding_model(),
    }


def main():
    st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💬", layout="wide")

    st.markdown(
        "<style>"
        "body {background-color: #f7f7f8;}"
        ".chat-bubble {padding: 14px; border-radius: 18px; margin: 8px 0; max-width: 84%; line-height: 1.5;}"
        ".user {background: #daf8cb; margin-left: auto; color: #0b3d1b;}"
        ".assistant {background: #ffffff; margin-right: auto; color: #2b2b2b;}"
        ".chat-container {display: flex; flex-direction: column; gap: 10px;}"
        ".chat-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}"
        ".chat-subtitle {color: #4d4d4d; margin-top: 0; margin-bottom: 1.5rem;}"
        ".streamlit-expanderHeader {display: none;}"
        "</style>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chat-title'>Zyro Dynamics HR Help Desk</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='chat-subtitle'>Ask HR policy questions and get answers directly from the company documents."
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        resources = load_rag_components()
    except Exception as error:
        st.error(f"Failed to load the RAG system: {error}")
        return

    if "history" not in st.session_state:
        st.session_state.history = []

    if hasattr(st, "chat_input"):
        user_message = st.chat_input("Send a message")
    else:
        user_message = st.text_area("Send a message", height=100)

    if user_message:
        with st.spinner("Finding the best answer in HR policies..."):
            answer = resources["ask"](user_message)
        st.session_state.history.append({"role": "user", "message": user_message})
        st.session_state.history.append({"role": "assistant", "message": answer})

    if st.session_state.history:
        for entry in st.session_state.history:
            if entry["role"] == "user":
                st.markdown(
                    f"<div class='chat-container'><div class='chat-bubble user'><strong>You</strong><br>{entry['message']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-container'><div class='chat-bubble assistant'><strong>Zyro HR Assistant</strong><br>{entry['message']}</div></div>",
                    unsafe_allow_html=True,
                )

    st.write("---")
    st.markdown(
        "**Tip:** Ask about leave policy, work from home, compensation and benefits, performance reviews, travel and expense, onboarding, separation, or code of conduct."
    )


if __name__ == "__main__":
    main()
