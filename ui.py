import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Medical Assistant RAG", page_icon="🩺")

st.title("🩺 Medical Assistant RAG")
st.markdown("Ask a question about the medical documents.")

# Input
question = st.text_input("Your Question", placeholder="e.g., What are the symptoms of Influenza?")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Consulting the medical database..."):
            try:
                # Send request to FastAPI backend
                payload = {"question": question}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    sources = data.get("source_documents", [])
                    time_taken = data.get("total_time", "N/A")

                    # Display Answer
                    st.success("Answer Generated!")
                    st.markdown(f"**Answer:**")
                    st.write(answer)
                    
                    st.caption(f"Processing Speed: {time_taken}")

                    # Display Sources
                    if sources:
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1}**")
                                st.markdown(f"*File: {doc.get('metadata', {}).get('source', 'Unknown')}*")
                                st.text(doc.get("page_content", "")[:500] + "...") # Preview
                                st.divider()
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server. Is `main.py` running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a RAG (Retrieval Augmented Generation) pipeline "
    "to answer medical questions based on the provided PDF documents."
)
