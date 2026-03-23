# Simple Streamlit Chatbot

This is a simple chatbot application using Streamlit for the frontend and OpenAI's Chat API for the backend. It now supports multiple knowledge sources (OFAC SDN Enhanced and US HTS CSVs) via a reuseable RAG index.

## Configuration

This application is pre-configured for Azure OpenAI: the code sets `openai.api_type = "azure"`. To get started, export the following environment variables:

```bash
# Azure OpenAI resource endpoint (no path suffix)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# (Optional) Azure OpenAI API version; defaults to 2025-04-01-preview if not set
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Your Azure OpenAI API key
export AZURE_OPENAI_API_KEY="your_api_key_here"

# The deployment name for your model (e.g. gpt-5-mini)
export AZURE_OPENAI_DEPLOYMENT_ID="gpt-5-mini"
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

## Knowledge sources

- Drop `SDN_ENHANCED.ZIP` into the repo root to use the OFAC SDN Enhanced XML archive.
- Place one or more HTS CSV exports that match `hts_*_csv.csv` (for example `hts_2026_revision_4_csv.csv`). When multiple files exist, the app automatically picks the highest revision (and most recent modification time) so future revisions can be added in place.
- The sidebar lets you choose between the available sources before issuing a query; the RAG cache is scoped per file, so replacing `hts_2026_revision_4_csv.csv` with `hts_2026_revision_5_csv.csv` simply triggers cache recomputation when you rerun the app.

Happy chatting!
**macOS/WSL users**: run these commands in a bash/zsh shell on macOS or within WSL; on Debian/Ubuntu you may need to install the `python3-venv` package (e.g. `sudo apt install python3-venv`) before creating the environment.
