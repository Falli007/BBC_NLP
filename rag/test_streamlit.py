import streamlit as st

st.title("Test Streamlit App")
st.write("If you can see this, Streamlit is working!")

# Test if we can import our RAG system
try:
    import sys
    sys.path.append('..')
    from rag_system import BBCRAGSystem
    st.success("✅ RAG system imports successfully!")
except Exception as e:
    st.error(f"RAG system import failed: {str(e)}")

st.write("API Key Status:", "Set" if st.secrets.get("OPENAI_API_KEY") else "Not set")
