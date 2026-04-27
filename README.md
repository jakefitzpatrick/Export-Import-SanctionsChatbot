# Import Insight AI

This is a Streamlit chatbot that translates natural-language questions into SQL and runs them deterministically against a local copy of the U.S. Harmonized Tariff Schedule (HTS). It pairs those results with governance-risk scores so every response is grounded in the same repeatable dataset.

## Key Features

- **Deterministic SQL execution:** Natural-language questions are converted to SELECT statements that run against `data/hts.db` (specifically the `hts_with_ch99` view so Chapter 99 surcharges are always factored in).
- **Risk & tariff analysis:** Pick up to three countries and one HTS code, click **Analyze**, and receive a Plotly scatter plot, data table, and AI-generated narrative—plus warnings for quantity-based duties.
- **Find Best Countries:** Click **✦ Find Best Countries** after selecting an HTS code to let the AI automatically recommend the three optimal sourcing markets, ranked by a weighted mix of tariff rate, corruption score, and trade-flow reasoning.
- **Specific duty support:** HTS codes with per-unit rates (e.g. 68¢/head) are handled separately. A **⊞ Specific Duty** scatter tab plots each country's per-unit rate against its corruption score, and Ch.99 ad-valorem surcharges (e.g. Section 122) that cannot be applied to a per-unit rate are flagged as a note in the data table.
- **Chapter 99 surcharge layering:** The `chapter_99` table is joined to base HTS rates via the `hts_with_ch99` view. Floor rates, additive surcharges, and trade-program overrides (USMCA, CAFTA-DR, Section 122, etc.) are resolved by match priority and applied per country.
- **Context-aware follow ups:** The chatbot automatically cites the latest analysis bundle (selections, risk snapshot, Chapter 99 summary, tariff breakdown) when answering follow-up questions.
- **Special program lookup:** Type `/program CODE` (e.g., `/program A+` or `/program KR`) to get a plain-language description of the special tariff program symbols shown in the HTS “Special” column.
- **Guaranteed priority HTS codes:** The product dropdown lists the first 1,000 HTS codes for snappy performance and always includes `0406.40.44.00` and `0405.90.20` even if they fall outside that slice.

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
python scripts/build_hts_sqlite.py \
  --csv hts_cleaned_final.csv \
  --ch99 Final_Wildcard_Optimized_Chapter99.csv \
  --db data/hts.db
streamlit run app.py
```

> **Important:** Keep `hts_cleaned_final.csv` and `Final_Wildcard_Optimized_Chapter99.csv` in this directory so the build script can populate both the `hts` table and the `chapter_99` table in `data/hts.db`. Every query a user submits is translated by Azure OpenAI into SQL and executed against the local database for a repeatable answer set.

Happy querying!

**macOS/WSL users:** run these commands in a bash/zsh shell on macOS or within WSL; on Debian/Ubuntu you may need to install the `python3-venv` package (e.g. `sudo apt install python3-venv`) before creating the environment.

## Chat shortcuts

- `/program CODE` — returns the description of a special tariff program symbol (e.g., `/program S+` explains the USMCA designation).
- **✦ Find Best Countries** — appears in the sidebar after selecting an HTS code; triggers an AI-powered country ranking without requiring manual country selection.
- Suggestion chips under each analysis bubble auto-populate the chat input with common follow-up questions (risk checks, lowest-duty country, trade-programs) so you can keep digging without retyping.
