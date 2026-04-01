from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

retriever = None

def get_vector_store_name(db_type: str, retriver_k: int = 10) -> Chroma:
    """Initialize and set the global retriever for the given db_type."""
    global retriever
    if db_type == "small":
        DB_NAME = str(Path(__file__).parent.parent / "vector_db_small")
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    elif db_type == "large":
        DB_NAME = str(Path(__file__).parent.parent / "vector_db_large")
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    elif db_type == "hybrid":
        DB_NAME = str(Path(__file__).parent.parent / "vector_db_hybrid")
        vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        raise ValueError("Invalid db_type. Must be 'small', 'large', or 'hybrid'.")

    # Update the global retriever to use the newly initialized vector store
    retriever = vectorstore.as_retriever(k=retriver_k)
    return vectorstore

# Provide a default retriever in case get_vector_store_name is not called
get_vector_store_name("small")

llm = ChatOpenAI(temperature=0, model_name=MODEL)

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))  ## Need to convert the openAI chat history ("user", "assistant") to LangChain format (SystemMessage, HumanMessage)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
