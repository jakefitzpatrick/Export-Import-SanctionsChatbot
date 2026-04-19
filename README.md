# Import Insight AI

This is a Streamlit chatbot that translates natural-language questions into SQL and runs them deterministically against a local copy of the U.S. Harmonized Tariff Schedule (HTS).

## Configuration

Configure your Azure OpenAI credentials so the app can request SQL from your deployment:

```bash
# Azure OpenAI resource endpoint (no path suffix)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# (Optional) Azure OpenAI API version; defaults to 2025-04-01-preview if not set
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# Your Azure OpenAI API key
export AZURE_OPENAI_API_KEY="your_api_key_here"

# The deployment name for your chat-capable model (e.g. gpt-5-mini)
export AZURE_OPENAI_DEPLOYMENT_ID="gpt-5-mini"
```

No embedding deployment is required because every prompt is converted into a SQL statement that is executed locally.

## Launching

1. Create and activate a virtual environment (if you have not already):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies into the virtual environment:

```bash
pip install -r requirements.txt
```

3. Build the SQLite database and run the Streamlit app:

```bash
python scripts/build_hts_sqlite.py
streamlit run app.py
```

> **Important:** Keep `hts_cleaned_final.csv` (included in the repo) in this directory so the build script can refresh `data/hts.db`. Every query a user submits is translated by Azure OpenAI into SQL and executed against the local database for a repeatable answer set.

Happy querying!

**macOS/WSL users:** run these commands in a bash/zsh shell on macOS or within WSL; on Debian/Ubuntu you may need to install the `python3-venv` package (e.g. `sudo apt install python3-venv`) before creating the environment.
