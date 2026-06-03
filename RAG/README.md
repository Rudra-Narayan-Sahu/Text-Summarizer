# RAG via WebSearch

A Streamlit application that implements a Retrieval-Augmented Generation (RAG) system using LangChain, Groq, and Ollama embeddings.

## Prerequisites

1. **Groq API Key**: Get from [https://console.groq.com](https://console.groq.com)
2. **Ollama**: Install from [https://ollama.ai](https://ollama.ai) and have it running locally
3. **Python 3.8+**

## Setup

1. **Download the required Ollama model**:
   ```bash
   ollama pull nomic-embed-text
   ```

2. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

3. **Activate the virtual environment**:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

2. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

## Features

- Loads content from Wikipedia
- Splits documents into manageable chunks
- Creates embeddings using Ollama's nomic-embed-text model
- Uses Groq's Llama 3.1 8B model for fast inference
- Interactive Q&A interface with Streamlit

## Troubleshooting

- **"GROQ_API_KEY not found"**: Make sure you have a `.env` file with your API key
- **"Connection refused" for Ollama**: Make sure Ollama is running (`ollama serve`)
- **Import errors**: Verify all packages are installed with `pip install -r requirements.txt`
