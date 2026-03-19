"""Streamlit sandbox-aware chatbot that references the SDN list via RAG."""
import os
from pathlib import Path

import openai
import streamlit as st

from rag import DEFAULT_EMBEDDING_MODEL, RagIndex

RAG_ARCHIVE = Path("SDN_ENHANCED.ZIP")
DEFAULT_TOP_K = 3


@st.cache_resource(show_spinner=False)
def load_rag_index(embedding_model: str) -> RagIndex:
    index = RagIndex(archive_path=RAG_ARCHIVE)
    index.ensure_index(embedding_model)
    return index


def _format_reference_sections(results: list[dict]) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        metadata = item["metadata"] or {}
        entity_name = metadata.get("entity_name") or "Unknown entity"
        entity_id = metadata.get("entity_id", "N/A")
        lines.append(
            f"{idx}. {entity_name} (ID {entity_id})\nScore: {item['score']:.3f}\n{item['chunk']}"
        )
    return "\n\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="Sanctions RAG Assistant", layout="wide")
    st.title("Sanctions RAG Assistant")
    st.caption("Uses the OFAC SDN ENHANCED XML archive as the knowledge source.")

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        st.error("Please set AZURE_OPENAI_API_KEY before running the app.")
        return
    if not openai.api_base:
        st.error("Please set AZURE_OPENAI_ENDPOINT before running the app.")
        return
    openai.api_key = api_key

    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    if not deployment_id:
        st.error("Please set AZURE_OPENAI_DEPLOYMENT_ID for the chat completion deployment.")
        return

    if not RAG_ARCHIVE.exists():
        st.warning(
            "Please drop SDN_ENHANCED.ZIP into the app root so the RAG index can be built."
        )
        return

    embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", DEFAULT_EMBEDDING_MODEL
    )

    try:
        rag_index = load_rag_index(embedding_deployment)
    except Exception as exc:
        st.error(f"Failed to prepare the RAG index: {exc}")
        return

    with st.sidebar:
        st.header("Retrieval knobs")
        top_k = st.slider("Chunks to consider", 1, 5, DEFAULT_TOP_K)
        st.caption(
            "Searches SDN entities for semantic matches, then passes the highest scoring chunks to the chat completion."
        )
        st.caption(
            "Temperature is fixed at 1.0 because this deployment currently only supports the default value."
        )

    question = st.text_area(
        "Ask about the SDN Enhanced List (e.g., sanctioned countries, aliases, digital assets):",
        height=140,
        key="prompt_input",
    )
    if st.button("Send query to Azure OpenAI", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before sending it to Azure OpenAI.")
            return

        reference_chunks = rag_index.search(question, top_k=top_k)
        if reference_chunks:
            context_text = _format_reference_sections(reference_chunks)
            with st.expander("Validation snippets from SDN_ENHANCED", expanded=True):
                st.text_area(
                    "Top matches", value=context_text, height=240, max_chars=None, key="doc_view"
                )
        else:
            context_text = ""
            st.info("No semantically similar documents were found in the archive.")

        messages = [
            {
                "role": "system",
                "content": "You answer questions using the OFAC SDN Enhanced data; cite relevant retrieved evidence when available.",
            }
        ]
        if context_text:
            messages.append(
                {
                    "role": "system",
                    "content": "Reference data:\\n" + context_text,
                }
            )
        messages.append({"role": "user", "content": question})

        with st.spinner("Sending request to Azure OpenAI chat deployment..."):
            response = openai.chat.completions.create(
                model=deployment_id,
                messages=messages,
                temperature=1.0,
            )
        reply = response.choices[0].message.content
        st.text_area("Bot response", value=reply, height=220)


if __name__ == "__main__":
    main()
