# RAG Agent for Q&A on Restaurant Reviews Dataset

A Retrieval-Augmented Generation (RAG) system that answers questions about a pizza restaurant using customer reviews. (used as basis for larger business-focuesed RAG system)

## Features
- Semantic search using vector embeddings
- Local LLM integration with Ollama
- Persistent vector database with Chroma

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and pull required models:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull mxbai-embed-large
```

3. Run the application:
```bash
python main.py
```

## Project Structure
- `main.py`: Main application and Q&A interface
- `vector.py`: Vector database and retrieval setup
- `requirements.txt`: Project dependencies

## Dependencies
- langchain
- langchain-ollama
- langchain-chroma
- pandas
- Ollama (local LLM) 
