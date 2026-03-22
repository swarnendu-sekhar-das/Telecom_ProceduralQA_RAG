import streamlit as st
import os
import sys

# Ensure the src module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm.qa_engine import ProceduralQAEngine

# Configure the Streamlit page
st.set_page_config(
    page_title="SecureNOC QA RAG",
    page_icon="📡",
    layout="wide",
)

# Initialize the QA Engine (Now just an API Client)
@st.cache_resource
def load_qa_engine():
    return ProceduralQAEngine()

# Header Information
st.title("📡 SecureNOC: Procedural QA RAG")
st.markdown("""
This system uses a decoupled Retrieval-Augmented Generation (RAG) pipeline to find and output exact Standard Operating Procedures (SOPs).
All embeddings, hybrid retrieval, reranking, and LLM generation are served via a secure Colab API endpoint.
""")

qa_engine = load_qa_engine()

st.divider()

# Sidebar: Stripped down since GLiNER handles metadata filtering automatically
st.sidebar.header("System Status")
st.sidebar.success("Ready to connect to SecureNOC API")
st.sidebar.markdown("""
**Note:** Metadata pre-filtering (like Equipment Vendor or Severity) is now handled *automatically* by Zero-Shot NLP (GLiNER) directly from your prompt text.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a clear chat button to the sidebar
if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Show Retrieved SOPs (Cross-Encoder Top 3)"):
                for source in message["sources"]:
                    st.markdown(f"**SOP ID:** `{source['sop_id']}` | **Vendor:** `{source['vendor']}` | **Severity:** `{source['severity']}`")
                    st.text(f"Title: {source['title']}")

# React to user input
if prompt := st.chat_input("Ask a procedural question (e.g., 'How do I fix a corrupted BGP table on a Cisco router?'):"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
        
    # Query Engine
    with st.chat_message("assistant"):
        with st.spinner("Pinging SecureNOC API (Retrieving & Synthesizing)..."):
            try:
                # We no longer pass manual filters; the backend handles it.
                response = qa_engine.query(prompt)
                st.markdown(response.response)
                
                # Extract Sources explicitly for the UI based on the new MockNode structure
                source_data = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("Show Retrieved SOPs (Cross-Encoder Top 3)"):
                        for node in response.source_nodes:
                            # Read directly from the MockNode's metadata dictionary
                            sop_id = node.metadata.get('sop_id', 'Unknown')
                            vendor = node.metadata.get('vendor', 'Unknown')
                            severity = node.metadata.get('severity', 'UNKNOWN')
                            title = node.metadata.get('title', 'Untitled')
                            
                            st.markdown(f"**SOP ID:** `{sop_id}` | **Vendor:** `{vendor}` | **Severity:** `{severity}`")
                            st.text(f"Title: {title}")
                            
                            source_data.append({
                                "sop_id": sop_id,
                                "vendor": vendor,
                                "severity": severity,
                                "title": title
                            })
                            
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.response, "sources": source_data})
                
            except Exception as e:
                error_msg = f"**Error:** Failed to reach the backend API. Ensure the Colab Ngrok tunnel is running.\n\n`{str(e)}`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})