import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Medical Assistant RAG", page_icon="🩺", layout="wide")

# Custom CSS for better chat bubbles and source styling
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-doc {
        background-color: #f7f9fc;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Medical Assistant RAG")
st.markdown("Ask a question about the medical documents.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

import os

# Sidebar for controls and info
with st.sidebar:
    st.header("About")
    st.info(
        "This application uses a RAG (Retrieval Augmented Generation) pipeline "
        "to answer medical questions based on the provided PDF documents."
    )
    
    st.divider()
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            # 1. Save file to data directory
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # 2. Trigger ingestion and stream progress
            with st.status("Ingesting Document...", expanded=True) as status:
                try:
                    # Point to the new ingest endpoint
                    response = requests.post(API_URL.replace("/query", "/ingest"), stream=True)
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                st.write(decoded_line)
                        status.update(label="Ingestion Complete!", state="complete", expanded=False)
                        st.success(f"Successfully processed {uploaded_file.name}")
                    else:
                        status.update(label="Ingestion Failed", state="error")
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    status.update(label="Ingestion Error", state="error")
                    st.error(f"Connection failed: {e}")
                    
    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render sources if they exist for the assistant message
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Source Documents"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}**")
                    st.markdown(f"*File: {doc.get('metadata', {}).get('source', 'Unknown')}*")
                    # Display preview securely with line breaks
                    st.text(doc.get("page_content", "")[:500] + "...")
                    st.divider()

# Accept new user input
if prompt := st.chat_input("e.g., What are the symptoms of Influenza?"):
    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Fetch assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the medical database..."):
            try:
                payload = {"question": prompt}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("answer", "No answer found.")
                    sources = data.get("source_documents", [])
                    time_taken = data.get("total_time", "N/A")
                    
                    # Display the answer
                    st.markdown(answer_text)
                    st.caption(f"Processing Speed: {time_taken}")
                    
                    # Store in session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "sources": sources
                    })
                    
                    # Display sources immediately for the current block
                    if sources:
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1}**")
                                st.markdown(f"*File: {doc.get('metadata', {}).get('source', 'Unknown')}*")
                                st.text(doc.get("page_content", "")[:500] + "...")
                                st.divider()
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to the backend server. Is the API running?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
