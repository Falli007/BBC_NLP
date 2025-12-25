"""
BBC News RAG System

This module implements a Retrieval-Augmented Generation (RAG) system for the BBC news dataset.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            for i in range(min(100, chunk_size)):
                if text[end - i] in '.!?':
                    end = end - i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text) - overlap:
            break
    
    return chunks

def preprocess_articles(df: pd.DataFrame, chunk_size: int = 500) -> List[Dict]:
    """
    Preprocess articles and create chunks for RAG.
    
    Returns:
        List of dictionaries containing chunk information
    """
    processed_chunks = []
    
    for idx, row in df.iterrows():
        text = row['Text']
        category = row['Category']
        filename = row.get('Filename', f'article_{idx}')
        
        # Create chunks
        chunks = chunk_text(text, chunk_size=chunk_size)
        
        for chunk_idx, chunk in enumerate(chunks):
            processed_chunks.append({
                'article_id': idx,
                'chunk_id': f"{idx}_{chunk_idx}",
                'text': chunk,
                'category': category,
                'filename': filename,
                'chunk_length': len(chunk)
            })
    
    return processed_chunks

def create_embeddings_and_index(chunks: List[Dict], model: SentenceTransformer) -> Tuple[np.ndarray, faiss.Index, List[Dict]]:
    """
    Create embeddings for all chunks and build FAISS index.
    
    Returns:
        embeddings: numpy array of embeddings
        index: FAISS index for similarity search
        chunk_metadata: metadata for each chunk
    """
    print(f"Creating embeddings for {len(chunks)} chunks...")
    
    # Extract text for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    # Create embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    print(f"SUCCESS: Created FAISS index with {index.ntotal} vectors")
    
    return embeddings, index, chunks

def retrieve_relevant_chunks(question: str, index: faiss.Index, model: SentenceTransformer, 
                           chunk_metadata: List[Dict], k: int = 5) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a given question.
    
    Args:
        question: User's question
        index: FAISS index
        model: Sentence transformer model
        chunk_metadata: Metadata for each chunk
        k: Number of chunks to retrieve
    
    Returns:
        List of relevant chunks with metadata
    """
    # Create embedding for the question
    question_embedding = model.encode([question])
    faiss.normalize_L2(question_embedding)
    
    # Search for similar chunks
    scores, indices = index.search(question_embedding.astype('float32'), k)
    
    # Get relevant chunks
    relevant_chunks = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk_info = chunk_metadata[idx].copy()
        chunk_info['similarity_score'] = float(score)
        chunk_info['rank'] = i + 1
        relevant_chunks.append(chunk_info)
    
    return relevant_chunks

def format_context_for_llm(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into context for the LLM.
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}] (Category: {chunk['category']}, Score: {chunk['similarity_score']:.3f})")
        context_parts.append(chunk['text'])
        context_parts.append("\n")
    
    return "\n".join(context_parts)

def generate_answer(question: str, context: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generate an answer using OpenAI API with retrieved context.
    
    Args:
        question: User's question
        context: Retrieved context from documents
        model: OpenAI model to use
    
    Returns:
        Generated answer
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not client:
        return "Error: OpenAI API client not available. Please set your API key."
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from BBC news articles.

Context from BBC News Articles:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so
- Be specific and cite relevant details from the context
- If there are multiple sources, mention the different perspectives
- Keep your answer concise but informative

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on BBC news articles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

class BBCRAGSystem:
    """
    Complete RAG system for BBC news articles.
    """
    
    def __init__(self, df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        self.df = df
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_metadata = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    
    def build_index(self, chunk_size: int = 500, sample_size: int = None):
        """
        Build the vector index from articles.
        """
        print(f"Building RAG index...")
        
        # Use sample if specified
        df_to_process = self.df.head(sample_size) if sample_size else self.df
        
        # Create chunks
        chunks = preprocess_articles(df_to_process, chunk_size=chunk_size)
        print(f"Created {len(chunks)} chunks from {len(df_to_process)} articles")
        
        # Create embeddings and index
        embeddings, index, chunk_metadata = create_embeddings_and_index(chunks, self.model)
        
        self.index = index
        self.chunk_metadata = chunk_metadata
        
        print(f"SUCCESS: RAG index built successfully!")
        
    def ask(self, question: str, k: int = 5, model: str = "gpt-3.5-turbo") -> Dict:
        """
        Ask a question and get an answer using RAG.
        
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if self.index is None:
            return {"error": "Index not built. Please call build_index() first."}
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            question, self.index, self.model, self.chunk_metadata, k=k
        )
        
        # Format context
        context = format_context_for_llm(relevant_chunks)
        
        # Generate answer
        if self.client:
            answer = generate_answer(question, context, model)
        else:
            answer = "OpenAI API not available. Please set your API key."
        
        return {
            "question": question,
            "answer": answer,
            "sources": relevant_chunks,
            "context": context,
            "num_sources": len(relevant_chunks)
        }
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk.
        """
        if self.index is None:
            print("No index to save.")
            return
        
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load the FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            self.chunk_metadata = json.load(f)
        
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
