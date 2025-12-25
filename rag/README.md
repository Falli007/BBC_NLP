# BBC News RAG System

A Retrieval-Augmented Generation (RAG) system for querying BBC news articles using AI-powered search and answer generation.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

### 3. Launch the Streamlit App

```bash
python run_streamlit.py
```

Or directly with Streamlit:

```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
rag/
├── rag_system.ipynb      # Jupyter notebook with RAG implementation
├── streamlit_app.py      # Streamlit web interface
├── run_streamlit.py      # Launcher script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Features

### Web Interface (Streamlit)
- **Interactive Q&A**: Ask questions about BBC news articles
- **Real-time Search**: Fast vector-based retrieval
- **Source Attribution**: See which articles were used for answers
- **Analytics Dashboard**: Visualize dataset statistics
- **Configurable Settings**: Adjust model, sources, and dataset size

### Jupyter Notebook
- **Educational**: Step-by-step implementation
- **Experimentation**: Test different parameters
- **Debugging**: Detailed logging and error handling
- **Batch Processing**: Process full dataset

## 🛠️ Technical Details

### Architecture
1. **Document Processing**: Split articles into chunks with smart sentence boundaries
2. **Vector Embeddings**: Use sentence-transformers (all-MiniLM-L6-v2)
3. **Vector Store**: FAISS index for fast similarity search
4. **Retrieval**: Find most relevant chunks for user questions
5. **Generation**: Use OpenAI GPT models to generate answers

### Models Used
- **Embedding**: `all-MiniLM-L6-v2` (384 dimensions)
- **Generation**: `gpt-3.5-turbo` or `gpt-4`
- **Vector Search**: FAISS with cosine similarity

### Dataset
- **Source**: BBC News articles (2,225 articles)
- **Categories**: Business, Entertainment, Politics, Sport, Tech
- **Chunking**: 400-500 character chunks with 50 character overlap

## 📊 Usage Examples

### Web Interface
1. Open the Streamlit app
2. Choose dataset size (Sample or Full)
3. Enter your question
4. View the AI-generated answer with sources

### Programmatic Usage
```python
from rag_system import BBCRAGSystem
import pandas as pd

# Load data
df = pd.read_csv("../data/raw_bbc.csv")

# Initialize RAG system
rag = BBCRAGSystem(df)
rag.build_index(sample_size=50)

# Ask questions
result = rag.ask("What happened with Time Warner profits?")
print(result['answer'])
```

## ⚙️ Configuration

### Streamlit Settings
- **Dataset Size**: Sample (50 articles) or Full (2,225 articles)
- **Number of Sources**: 1-10 relevant chunks to retrieve
- **OpenAI Model**: gpt-3.5-turbo or gpt-4

### RAG Parameters
- **Chunk Size**: 400-500 characters
- **Overlap**: 50 characters
- **Similarity Threshold**: Configurable in retrieval

## 🔍 Example Questions

- "What happened with Time Warner profits?"
- "What are the latest developments in technology?"
- "What sports news is there?"
- "What political events are happening?"
- "What entertainment news is there?"

## 📈 Performance

### Sample Dataset (50 articles)
- **Index Build Time**: ~30 seconds
- **Query Response**: ~2-5 seconds
- **Memory Usage**: ~100MB

### Full Dataset (2,225 articles)
- **Index Build Time**: ~10-20 minutes
- **Query Response**: ~2-5 seconds
- **Memory Usage**: ~500MB

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   - Ensure the environment variable is set correctly
   - Check that the key is valid and has credits

2. **Import Errors**
   - Make sure all dependencies are installed
   - Check that the BBC dataset exists at `../data/raw_bbc.csv`

3. **Memory Issues**
   - Use the sample dataset for testing
   - Close other applications to free up memory

4. **Slow Performance**
   - Use the sample dataset for faster responses
   - Reduce the number of sources retrieved

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- BBC for the news dataset
- OpenAI for the GPT models
- Hugging Face for sentence-transformers
- Facebook AI Research for FAISS
- Streamlit for the web framework

