# Export-Import Sanctions Chatbot

This is a simple chatbot application using Streamlit for the frontend and OpenAI's Chat API for the backend.

## Configuration

This application is pre-configured for Azure OpenAI and uses the U.S. Harmonized Tariff Schedule (HTS) dataset as its retrieval corpus before calling chat completions. To get started, export the following environment variables:

```bash
# Azure OpenAI resource endpoint (no path suffix)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# (Optional) Azure OpenAI API version; defaults to 2025-04-01-preview if not set
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Your Azure OpenAI API key
export AZURE_OPENAI_API_KEY="your_api_key_here"

# The deployment name for your model (e.g. gpt-5-mini)
export AZURE_OPENAI_DEPLOYMENT_ID="gpt-5-mini"

# Embeddings deployment must resolve to a deployment that exposes `text-embedding-3-large`;
# the default is `"text-embedding-3-large"` but you can override the deployment name via an env var.
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID="text-embedding-3-large"
```

## Launching

1. Create and activate a virtual environment (if not already):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies into the virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app from within the activated environment:

   ```bash
   streamlit run app.py
   ```

   > **Important:** Keep `hts_2026_revision_4_csv.csv` (included in the repo) in this directory so the RAG index can read the HTS tariff rows; the first run still needs to chunk and embed the data, so expect a few minutes for initialization depending on your embedding quota.

Happy chatting!
**macOS/WSL users**: run these commands in a bash/zsh shell on macOS or within WSL; on Debian/Ubuntu you may need to install the `python3-venv` package (e.g. `sudo apt install python3-venv`) before creating the environment.
