import streamlit as st
import os
import sys
import time
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm.qa_engine import ProceduralQAEngine, StreamingResponse

# Configure the Streamlit page
st.set_page_config(
    page_title="NetRestore: Procedural QA RAG",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom streaming animation */
        .streaming-cursor {
            display: inline-block;
            width: 0.1em;
            height: 1.2em;
            background-color: #0066cc;
            animation: blink 1s infinite;
            margin-left: 2px;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        /* Enhanced chat styling */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Source styling */
        .source-container {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 0.25rem;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        
        /* Error styling */
        .error-message {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.25rem;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize the QA Engine with streaming enabled
@st.cache_resource
def load_qa_engine():
    return ProceduralQAEngine(enable_streaming=True, fallback_timeout=5)

# Enhanced session state management
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "streaming_complete" not in st.session_state:
        st.session_state.streaming_complete = True

init_session_state()

# Header Information
st.markdown("""
<h1 style='text-align: center;'>
    <span style='font-size: 60px;'>📡</span> 
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style='text-align: center; font-size: 45px;'>
    NetRestore: Procedural QA RAG
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 18px;'>
NetRestore is a RAG-based system designed for telecom network restoration operations by retrieving precise restoration SOPs for outages, faults, and incidents using hybrid search and LLM reasoning, enabling faster and more reliable service recovery.
</p>
""", unsafe_allow_html=True)

qa_engine = load_qa_engine()

st.divider()

st.sidebar.header("Settings")
st.sidebar.success("Ready to connect to NetRestore API")

# Streaming toggle
st.sidebar.toggle(
    "Enable Streaming",
    value=st.session_state.streaming_enabled,
    key="streaming_toggle",
    help="Toggle between streaming and non-streaming responses"
)

# Show sources toggle
st.sidebar.toggle(
    "Show Sources",
    value=st.session_state.show_sources,
    key="sources_toggle",
    help="Show/hide retrieved SOP sources"
)

# Clear chat button
if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.current_sources = []
    st.session_state.streaming_complete = True
    st.rerun()

# Display chat history with enhanced formatting
def display_message(message):
    """Display a single message with enhanced formatting"""
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    
    with st.chat_message(role):
        if role == "assistant":
            # Enhanced markdown rendering for assistant messages
            st.markdown(content, unsafe_allow_html=True)
            
            # Display sources if available and enabled
            if sources and st.session_state.show_sources:
                with st.expander("Retrieved SOPs (Cross-Encoder Top 3)", expanded=False):
                    for source in sources:
                        st.markdown(f"""
                        <div class="source-container">
                            <strong>SOP ID:</strong> <code>{source.get('sop_id', 'Unknown')}</code> | 
                            <strong>Vendor:</strong> <code>{source.get('vendor', 'Unknown')}</code> | 
                            <strong>Severity:</strong> <code>{source.get('severity', 'UNKNOWN')}</code><br>
                            <strong>Title:</strong> {source.get('title', 'Untitled')}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # User message
            st.markdown(content)

# Display all existing messages
for message in st.session_state.messages:
    display_message(message)

def display_streaming_response():
    """Handle streaming response display"""
    if not st.session_state.streaming_complete:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources_container = st.empty()
            current_sources = []
            
            try:
                # Get streaming response
                for chunk in qa_engine.query_stream(st.session_state.current_query):
                    if chunk.error:
                        st.error(f"**Streaming Error:** {chunk.error}")
                        break
                    
                    if chunk.sources:
                        current_sources = chunk.sources
                        st.session_state.current_sources = current_sources
                        
                        # Display sources if enabled
                        if st.session_state.show_sources and current_sources:
                            sources_html = ""
                            for source in current_sources:
                                sources_html += f"""
                                <div class="source-container">
                                    <strong>SOP ID:</strong> <code>{source.get('sop_id', 'Unknown')}</code> | 
                                    <strong>Vendor:</strong> <code>{source.get('vendor', 'Unknown')}</code> | 
                                    <strong>Severity:</strong> <code>{source.get('severity', 'UNKNOWN')}</code><br>
                                    <strong>Title:</strong> {source.get('title', 'Untitled')}
                                </div>
                                """
                            sources_container.markdown(sources_html, unsafe_allow_html=True)
                    
                    if chunk.content:
                        full_response += chunk.content
                        # Add streaming cursor
                        display_text = full_response + '<span class="streaming-cursor"></span>'
                        message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    
                    if chunk.complete:
                        # Remove cursor and finalize
                        message_placeholder.markdown(full_response)
                        st.session_state.streaming_complete = True
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response, 
                            "sources": current_sources
                        })
                        break
                        
            except Exception as e:
                st.error(f"**Error:** Failed to process streaming response: {str(e)}")
                st.session_state.streaming_complete = True

# Handle streaming display
display_streaming_response()

# Enhanced input handling
def handle_user_input(prompt):
    """Handle user input with streaming support"""
    # Display user message
    st.chat_message("user").markdown(prompt)
    
    # Add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set up streaming
    st.session_state.current_query = prompt
    st.session_state.streaming_complete = False
    
    # Trigger rerun to start streaming
    st.rerun()

# React to user input
if prompt := st.chat_input("Ask a procedural question (e.g., 'How do I fix a corrupted BGP table on a Cisco router?'):"):
    if st.session_state.streaming_enabled:
        handle_user_input(prompt)
    else:
        # Non-streaming mode (original behavior)
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
            
        # Query Engine (non-streaming)
        with st.chat_message("assistant"):
            with st.spinner("Pinging NetRestore API (Retrieving & Synthesizing)..."):
                try:
                    response = qa_engine.query(prompt)
                    st.markdown(response.response)
                    
                    # Extract Sources for UI
                    source_data = []
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        if st.session_state.show_sources:
                            with st.expander("📚 Retrieved SOPs (Cross-Encoder Top 3)", expanded=False):
                                for node in response.source_nodes:
                                    sop_id = node.metadata.get('sop_id', 'Unknown')
                                    vendor = node.metadata.get('vendor', 'Unknown')
                                    severity = node.metadata.get('severity', 'UNKNOWN')
                                    title = node.metadata.get('title', 'Untitled')
                                    
                                    st.markdown(f"""
                                    <div class="source-container">
                                        <strong>SOP ID:</strong> <code>{sop_id}</code> | 
                                        <strong>Vendor:</strong> <code>{vendor}</code> | 
                                        <strong>Severity:</strong> <code>{severity}</code><br>
                                        <strong>Title:</strong> {title}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    source_data.append({
                                        "sop_id": sop_id,
                                        "vendor": vendor,
                                        "severity": severity,
                                        "title": title
                                    })
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.response, 
                        "sources": source_data
                    })
                    
                except Exception as e:
                    error_msg = f"**Error:** Failed to reach the backend API. Ensure the Colab Ngrok tunnel is running.\n\n`{str(e)}`"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Performance info in sidebar
if st.session_state.messages:
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}")
    if st.session_state.streaming_enabled:
        st.sidebar.markdown("**Streaming:** Enabled")
    else:
        st.sidebar.markdown("**Streaming:** Disabled")