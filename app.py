import streamlit as st
import requests
import os
import zipfile
import shutil
import tempfile
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============ CONFIGURATION ============
# Securely load API key from Streamlit secrets
try:
    CHAT_API_KEY = st.secrets["CHAT_API_KEY"]
except KeyError:
    st.error("⚠️ API key not found. Please set CHAT_API_KEY in Streamlit secrets.")
    st.stop()

CHAT_API_URL = "https://hkust.azure-api.net/hkust-genai/v1/chat/completions"

# Configuration for zip download
ZIP_URL = st.secrets.get("ZIP_URL", "https://gohkust-my.sharepoint.com/:f:/r/personal/imaryan_ust_hk/Documents/docs?csf=1&web=1&e=p8KB41")
ZIP_NAME = "course_materials.zip"
EXTRACT_FOLDER = "course_materials"
DB_DIR = "./chroma_db"  # This will be created locally on Streamlit Cloud

# ============ DOWNLOAD AND EXTRACT FUNCTION ============
def download_and_extract_data():
    """Downloads the zip file and extracts it."""
    
    # Check if already extracted
    if os.path.exists(EXTRACT_FOLDER) and os.path.isdir(EXTRACT_FOLDER):
        if any(Path(EXTRACT_FOLDER).iterdir()):
            return EXTRACT_FOLDER
    
    try:
        # Download
        response = requests.get(ZIP_URL, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")
        
        with open(ZIP_NAME, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        # Extract
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
        
        # Clean up
        os.remove(ZIP_NAME)
        
        return EXTRACT_FOLDER
        
    except Exception as e:
        # Clean up if there's an error
        if os.path.exists(ZIP_NAME):
            os.remove(ZIP_NAME)
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER, ignore_errors=True)
        raise e

# ============ LOAD DOCUMENTS FROM EXTRACTED FOLDER ============
def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Recursively loads all text documents from the extracted folder into LangChain Documents."""
    documents = []
    folder_path = Path(folder_path)
    
    # Define which file extensions to process
    text_extensions = {'.txt', '.md', '.csv', '.json', '.py', '.js', '.html', '.css', '.xml', '.yaml', '.yml', '.rtf'}
    
    for file_path in folder_path.rglob('*'):
        if file_path.is_file():
            # Check if it's a text file
            if file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": str(file_path.relative_to(folder_path)),
                                    "file_name": file_path.name,
                                    "file_type": file_path.suffix
                                }
                            )
                            documents.append(doc)
                except Exception:
                    pass  # Silently skip files that can't be read
            
            # Add PDF support if needed
            elif file_path.suffix.lower() == '.pdf':
                try:
                    # You'll need to install PyPDF2 or pdfplumber
                    # import PyPDF2
                    # with open(file_path, 'rb') as f:
                    #     reader = PyPDF2.PdfReader(f)
                    #     content = ""
                    #     for page in reader.pages:
                    #         content += page.extract_text()
                    #     if content.strip():
                    #         doc = Document(
                    #             page_content=content,
                    #             metadata={
                    #                 "source": str(file_path.relative_to(folder_path)),
                    #                 "file_name": file_path.name,
                    #                 "file_type": file_path.suffix
                    #             }
                    #         )
                    #         documents.append(doc)
                    pass
                except Exception:
                    pass
    
    return documents

# ============ BUILD VECTOR DATABASE ============
def build_vector_database():
    """Build or load the vector database from extracted documents"""
    
    # Step 1: Download and extract
    extract_path = download_and_extract_data()
    
    if extract_path is None:
        return None, "Failed to download course materials"
    
    # Step 2: Check if vector DB already exists locally
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
            if vector_db._collection.count() > 0:
                return vector_db, "Loaded existing vector database"
        except Exception:
            pass  # If load fails, rebuild
    
    # Step 3: Build from extracted files
    try:
        # Load all documents
        documents = load_documents_from_folder(extract_path)
        
        if not documents:
            return None, "No valid documents found in course materials"
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create and persist vector database
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        vector_db.persist()
        
        return vector_db, f"Built vector database with {len(documents)} documents"
        
    except Exception as e:
        return None, f"Error building vector database: {str(e)}"

# ============ CUSTOM PROMPT TEMPLATE ============
SYSTEM_PROMPT = """You are a helpful teaching assistant for the ISOM3400 course. Your role is to help students understand course materials, clarify concepts, and guide them through their learning.

Instructions:
1. Provide a clear, comprehensive answer based ONLY on the course materials provided in the context
2. If the context doesn't contain the answer, politely say you don't have that information in the course materials
3. Use specific examples from the course materials when relevant
4. Organize your response in a clear, structured way
5. If explaining code, provide clear explanations with the code snippets
6. Be encouraging and supportive in your tone
7. Do NOT simply copy and paste the retrieved text - synthesize the information into a coherent response
8. If you're referencing specific course materials, mention them naturally (e.g., "In the lecture on X...")

Remember: Your goal is to help students understand the material, not just retrieve documents for them."""

# ============ CHAT RESPONSE FUNCTION ============
def get_chat_response(user_query: str, context: str) -> str:
    """Get response from HKUST Chat API"""
    
    headers = {
        "Content-Type": "application/json",
        "api-key": CHAT_API_KEY,
    }
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Based on the following course materials, please answer the student's question.

COURSE MATERIALS:
{context}

STUDENT QUESTION: {user_query}

Please provide a clear, synthesized answer based on the course materials above."""}
    ]
    
    data = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
        "messages": messages
    }
    
    response = requests.post(CHAT_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Chat API Error {response.status_code}: {response.text}")

# ============ INITIALIZE CHATBOT (Cached) ============
@st.cache_resource
def initialize_chatbot():
    """
    Initialize everything in the background - download, extract, build vector DB.
    This runs once and caches the result.
    """
    return build_vector_database()

# ============ STREAMLIT APP ============
st.set_page_config(page_title="ISOM3400 Assistant", page_icon="🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False

# ============ LOADING SCREEN ============
# Only show loading screen if chat is not ready
if not st.session_state.chat_ready:
    # Create a placeholder for the loading screen
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.title("📚 ISOM3400 Course Assistant")
        st.markdown("## 🚀 Initializing your course assistant...")
        st.markdown("---")
        st.markdown("**Please wait while the system loads course materials**")
        st.markdown("This may take 2-3 minutes on first launch.")
        st.markdown("")
        
        # Create a spinner that shows during initialization
        with st.spinner("Downloading course materials and building knowledge base..."):
            # Initialize the chatbot (this runs in the background)
            vector_db, status_message = initialize_chatbot()
        
        if vector_db is not None:
            st.session_state.vector_db = vector_db
            st.session_state.chat_ready = True
            # Clear the loading screen and rerun
            loading_placeholder.empty()
            st.rerun()
        else:
            # Show error and stop
            st.error(f"❌ Failed to initialize: {status_message}")
            st.stop()

# ============ MAIN APP (Only shown when chat is ready) ============
# Main chat interface
st.title("📚 ISOM3400 Course Assistant")
st.markdown("Ask questions about your course materials and get synthesized answers!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar with debug info
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Option to show debug info
    show_debug = st.checkbox("Show Debug Info", value=False)
    
    if show_debug:
        st.subheader("🔧 Debug Information")
        
        # Get status
        vector_db = st.session_state.get("vector_db")
        if vector_db is not None:
            st.success("✅ Vector database loaded")
            st.write(f"Document count: {vector_db._collection.count()}")
            
            if os.path.exists(EXTRACT_FOLDER):
                file_count = sum(1 for _ in Path(EXTRACT_FOLDER).rglob('*') if _.is_file())
                st.write(f"Files extracted: {file_count}")
        else:
            st.error("❌ Vector database not found")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ============ HANDLE USER INPUT ============
if user_query := st.chat_input("Ask a question about ISOM3400..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("💭 Analyzing course materials and generating response..."):
            try:
                vector_db = st.session_state.get("vector_db")
                
                if vector_db is None:
                    st.error("Vector database not available. Please restart the app.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Vector database not available."})
                else:
                    # Create retriever with MMR for diverse context
                    retriever = vector_db.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 6,
                            "fetch_k": 20,
                            "lambda_mult": 0.7
                        }
                    )
                    
                    # Retrieve relevant documents
                    docs = retriever.invoke(user_query)
                    
                    # Prepare context from retrieved documents
                    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                    
                    # Get synthesized response
                    answer = get_chat_response(user_query, context)
                    
                    # Display the answer
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show debug info if enabled
                    if show_debug:
                        with st.expander("🔍 Debug: Retrieved Sources"):
                            for i, doc in enumerate(docs):
                                source = doc.metadata.get("source", "Unknown")
                                st.write(f"**Source {i+1}:** {source}")
                                st.write(f"Preview: {doc.page_content[:150]}...")
                                st.divider()
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})