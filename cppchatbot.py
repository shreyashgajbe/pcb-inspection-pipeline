import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Set page configurations
st.set_page_config(
    page_title="C++ Assistant RAG App",
    page_icon="🤖",
    layout="centered"
)

# App Title & Description
st.title("🤖 C++ Interview & Knowledge Assistant")
st.markdown(
    "Ask any question regarding C++, and this app will perform a similarity search "
    "over the document corpus to fetch the most relevant answers."
)

# 1. Cache the entire Vector Database initialization pipeline
@st.cache_resource(show_spinner="Initializing Vector Database and processing text data...")
def initialize_vector_db(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: `{file_path}` file not found. Please make sure the text file is in the same directory.")
        st.stop()
        
    # Data Loading
    loader = TextLoader(file_path, encoding="utf-8")
    txt_data = loader.load()
    
    # Splitting Data into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    final_documents = text_splitter.split_documents(txt_data)
    
    # Data Embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Storage Creation
    vector_db = FAISS.from_documents(final_documents, embeddings)
    return vector_db

# Initialize the db using the source text file
DATA_FILE = "cppTextData.txt"
db = initialize_vector_db(DATA_FILE)

# Divider line
st.divider()

# 2. User Input Section
query = st.text_input(
    label="Enter your question for C++:", 
    placeholder="e.g., What is a class? or What is virtual constructor?",
)

# 3. Retrieval & Display Section
if query:
    with st.spinner("Searching for the most relevant answers..."):
        # Perform similarity search
        docs = db.similarity_search(query)
        
    if docs:
        st.subheader("🔍 Relevant Match Results:")
        
        # Display the results cleanly using expanders/cards
        for index, doc in enumerate(docs):
            with st.expander(f"Result #{index + 1}", expanded=(index == 0)):
                st.write(doc.page_content)
                # Display metadata if helpful
                st.caption(f"Source chunk from: {doc.metadata.get('source', 'Unknown')}")
    else:
        st.info("No matching content found for your query. Try rephrasing.")