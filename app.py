import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing import TypedDict, Annotated


class AgentState(TypedDict):
    question: str
    context: str
    answer: str
# Define retriever step
def retrieve_context(state: AgentState) -> AgentState:
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    # context = "\n\n".join([doc.page_content for doc in docs])
    context = "\n\n".join(
    [f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content.strip()}" for doc in docs])
    return {**state, "context": context}

# Define LLM response step (we'll plug in your local Ollama LLM next)
def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    prompt = f"Answer the question based only on this context:\n\n{context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return {**state, "answer": response}
# Streamlit config
st.set_page_config(page_title="LangGraph RAG App", layout="wide")
st.title("Ask Your Documents (LangGraph + Ollama)")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# Function to load documents
def load_documents(pdf_dir, csv_dir):
    all_docs = []

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            all_docs.extend(loader.load())

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            loader = CSVLoader(file_path=os.path.join(csv_dir, file))
            all_docs.extend(loader.load())

    return all_docs


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chunk and embed documents
def embed_documents(docs):
    texts = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(texts, embedding_model)  # embedding_model needs to be defined
    return vectorstore

# Load and embed documents
all_docs = load_documents(r"C:\sid\pdfs", r"C:\sid\csv")
vectorstore = embed_documents(all_docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Graph setup
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("retrieve_context", RunnableLambda(retrieve_context))
graph_builder.add_node("generate_answer", RunnableLambda(generate_answer))

# Define edges
graph_builder.set_entry_point("retrieve_context")
graph_builder.add_edge("retrieve_context", "generate_answer")
graph_builder.set_finish_point("generate_answer")

# Compile the graph
graph = graph_builder.compile()
llm = ChatOllama(model="llama3") 
template = """You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    RunnableMap({
        "context": retriever,  # Using the FAISS retriever
        "question": RunnablePassthrough()  # Pass the question along
    })
    | prompt  # Apply the prompt
    | llm  # Use LLM for generation
    | StrOutputParser()  # Parse the output
)
prompt = PromptTemplate.from_template(template)
# User input section
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    with st.spinner("Thinking..."):
        


        result = rag_chain.invoke(user_question)
        st.subheader(" Answer:")
        st.write(result)
