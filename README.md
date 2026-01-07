# fake-news-detector-agent

A LangGraph-based agent for detecting fake news in social media posts.

## Features

- Multi-model support (OpenAI, Ollama)
- Configurable model selection via environment variables
- Fact-checking workflow with research and analysis

## Setup

1. Install dependencies:
```bash
pip install langchain-openai langchain-ollama tavily-python python-dotenv
```

2. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL_NAME=gpt-3.5-turbo  # Optional, defaults to gpt-3.5-turbo
MODEL_TEMPERATURE=0  # Optional, defaults to 0
OLLAMA_BASE_URL=http://localhost:11434  # Optional, for Ollama models
```

## Usage

The project uses LangChain's built-in `init_chat_model` API, which supports multiple providers through a unified interface.

### Using OpenAI Models

```bash
# Set in .env or export
export MODEL_NAME=gpt-3.5-turbo
# or explicitly with provider prefix
export MODEL_NAME=openai:gpt-4

python main.py
```

### Using Ollama Models (Self-hosted)

1. Make sure Ollama is running locally:
```bash
ollama serve
```

2. Pull a model (if not already available):
```bash
ollama pull llama2
# or
ollama pull mistral
```

3. Run with Ollama model:
```bash
# Set in .env or export (must use provider prefix for Ollama)
export MODEL_NAME=ollama:llama2
# or
export MODEL_NAME=ollama:mistral

python main.py
```
