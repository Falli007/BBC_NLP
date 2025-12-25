import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our RAG system
import sys
sys.path.append('.')
from rag_system import BBCRAGSystem

# Page configuration
st.set_page_config(
    page_title="BBC News RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_bbc_data():
    """Load BBC dataset with caching."""
    try:
        df = pd.read_csv("../data/raw_bbc.csv")
        return df
    except FileNotFoundError:
        st.error("BBC dataset not found. Please ensure '../data/raw_bbc.csv' exists.")
        return None

@st.cache_resource
def initialize_rag_system(df, sample_size=None):
    """Initialize RAG system with caching."""
    if df is None:
        return None
    
    rag_system = BBCRAGSystem(df)
    
    # Build index
    with st.spinner("Building RAG index... This may take a few minutes."):
        rag_system.build_index(chunk_size=400, sample_size=sample_size)
    
    return rag_system

def display_metrics(rag_system):
    """Display system metrics."""
    if rag_system is None or rag_system.index is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", f"{rag_system.index.ntotal:,}")
    
    with col2:
        categories = set(chunk['category'] for chunk in rag_system.chunk_metadata)
        st.metric("Categories", len(categories))
    
    with col3:
        avg_length = np.mean([chunk['chunk_length'] for chunk in rag_system.chunk_metadata])
        st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
    
    with col4:
        articles = set(chunk['article_id'] for chunk in rag_system.chunk_metadata)
        st.metric("Articles Indexed", len(articles))

def display_category_distribution(rag_system):
    """Display category distribution chart."""
    if rag_system is None or rag_system.chunk_metadata is None:
        return
    
    # Count chunks by category
    category_counts = {}
    for chunk in rag_system.chunk_metadata:
        category = chunk['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Create pie chart
    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Chunks by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_sources(sources: List[Dict], max_display: int = 5):
    """Display retrieved sources in an attractive format."""
    if not sources:
        st.info("No sources found.")
        return
    
    st.subheader(f"Sources ({len(sources)})")
    
    for i, source in enumerate(sources[:max_display]):
        with st.expander(f"Source {i+1}: {source['category']} (Score: {source['similarity_score']:.3f})"):
            st.write(f"**Article ID:** {source['article_id']}")
            st.write(f"**Category:** {source['category']}")
            st.write(f"**Similarity Score:** {source['similarity_score']:.3f}")
            st.write(f"**Text:**")
            st.write(source['text'])
    
    if len(sources) > max_display:
        st.info(f"Showing top {max_display} sources. Total: {len(sources)}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📰 BBC News RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about BBC news articles using AI-powered retrieval and generation!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Dataset size selection
        dataset_size = st.selectbox(
            "Dataset Size",
            ["Sample (50 articles)", "Full Dataset (2,225 articles)"],
            help="Choose dataset size. Full dataset will take longer to process."
        )
        
        sample_size = 50 if "Sample" in dataset_size else None
        
        # Number of sources
        num_sources = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant sources to retrieve for each question"
        )
        
        # OpenAI model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4"],
            help="Choose the OpenAI model for answer generation"
        )
        
        # API key status
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("SUCCESS: OpenAI API key found")
        else:
            st.error("ERROR: OpenAI API key not found")
            st.info("Please set your API key: `$env:OPENAI_API_KEY = 'your-key'`")
    
    # Load data and initialize system
    df = load_bbc_data()
    if df is None:
        st.stop()
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state or st.session_state.get('sample_size') != sample_size:
        st.session_state.sample_size = sample_size
        st.session_state.rag_system = initialize_rag_system(df, sample_size)
    
    rag_system = st.session_state.rag_system
    
    if rag_system is None:
        st.error("Failed to initialize RAG system.")
        st.stop()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Ask Questions", "Analytics", "Browse Sources", "About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Ask Questions About BBC News</h2>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What happened with Time Warner profits?",
            help="Ask any question about the BBC news articles"
        )
        
        # Example questions
        st.markdown("**Example Questions:**")
        example_questions = [
            "What happened with Time Warner profits?",
            "What are the latest developments in technology?",
            "What sports news is there?",
            "What political events are happening?",
            "What entertainment news is there?"
        ]
        
        cols = st.columns(len(example_questions))
        for i, example in enumerate(example_questions):
            with cols[i]:
                if st.button(f"Q{i+1}", help=example):
                    st.session_state.example_question = example
        
        if 'example_question' in st.session_state:
            question = st.session_state.example_question
            del st.session_state.example_question
        
        # Process question
        if st.button("Search", type="primary") or question:
            if not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching for relevant information..."):
                    result = rag_system.ask(question, k=num_sources, model=model_choice)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Answer")
                    st.write(result['answer'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    display_sources(result['sources'])
                    
                    # Store result for other tabs
                    st.session_state.last_result = result
    
    with tab2:
        st.markdown('<h2 class="sub-header">System Analytics</h2>', unsafe_allow_html=True)
        
        # Metrics
        display_metrics(rag_system)
        
        # Category distribution
        st.markdown("### Category Distribution")
        display_category_distribution(rag_system)
        
        # Chunk length distribution
        if rag_system.chunk_metadata:
            chunk_lengths = [chunk['chunk_length'] for chunk in rag_system.chunk_metadata]
            
            fig = px.histogram(
                x=chunk_lengths,
                nbins=20,
                title="Chunk Length Distribution",
                labels={'x': 'Chunk Length (characters)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Browse Sources</h2>', unsafe_allow_html=True)
        
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            display_sources(result['sources'], max_display=10)
        else:
            st.info("Ask a question first to see sources.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About This System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How It Works
        
        This RAG (Retrieval-Augmented Generation) system combines:
        
        1. **Document Processing**: BBC news articles are split into chunks
        2. **Vector Search**: Chunks are converted to embeddings and stored in a FAISS index
        3. **Retrieval**: Your question is used to find the most relevant chunks
        4. **Generation**: An AI model generates answers based on the retrieved context
        
        ### Technical Details
        
        - **Embedding Model**: all-MiniLM-L6-v2 (sentence-transformers)
        - **Vector Store**: FAISS (Facebook AI Similarity Search)
        - **Generation**: OpenAI GPT models
        - **Chunking**: Smart sentence boundary detection
        
        ### Dataset
        
        - **Source**: BBC News articles
        - **Categories**: Business, Entertainment, Politics, Sport, Tech
        - **Total Articles**: 2,225
        - **Sample Size**: 50 articles (for testing)
        
        ### Features
        
        - Real-time question answering
        - Source attribution and scoring
        - Interactive analytics
        - Configurable parameters
        - Responsive design
        """)
        
        st.markdown("### 📝 Usage Tips")
        st.markdown("""
        - Start with the sample dataset for faster responses
        - Use specific questions for better results
        - Check the sources to verify information
        - Adjust the number of sources based on your needs
        """)

if __name__ == "__main__":
    main()
