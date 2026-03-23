"""Streamlit sandbox-aware chatbot that references multiple knowledge sources via RAG."""
import os
from pathlib import Path

import openai
import streamlit as st

from rag import (
    DEFAULT_EMBEDDING_MODEL,
    SOURCE_KIND_HTS,
    SOURCE_KIND_SDN,
    RagIndex,
    create_document_source,
    find_latest_hts_csv,
)

DEFAULT_TOP_K = 3


@st.cache_resource(show_spinner=False)
def load_rag_index(embedding_model: str, source_kind: str, source_path: str) -> RagIndex:
    document_source = create_document_source(source_kind, Path(source_path))
    index = RagIndex(document_source=document_source)
    index.ensure_index(embedding_model)
    return index


def _format_reference_sections(results: list[dict], source_kind: str) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        metadata = item["metadata"] or {}
        if source_kind == SOURCE_KIND_HTS:
            title = metadata.get("description") or metadata.get("hts_number") or "HTS chunk"
        else:
            title = metadata.get("entity_name") or metadata.get("primary_name") or "Unknown entity"
        identifier = metadata.get("entity_id") or metadata.get("row_number") or "N/A"
        lines.append(
            f"Source {idx}: {title} (ID {identifier})\nScore: {item['score']:.3f}\n{item['chunk']}"
        )
    return "\n\n".join(lines)


QUESTION_PLACEHOLDERS = {
    SOURCE_KIND_SDN: "Ask about the SDN Enhanced List (e.g., sanctioned countries, aliases, digital assets):",
    SOURCE_KIND_HTS: "Ask about HTS classifications, duty rates, or quotas (e.g., which products fall under a heading):",
}

GENERATION_SYSTEM_PROMPTS = {
    SOURCE_KIND_SDN: (
        "You are a regulatory compliance assistant that relies on the OFAC SDN Enhanced List. "
        "Always ground your answers in the retrieved evidence below, and cite the chunk number whenever you quote a finding (e.g., \"(Source 1)\"). "
        "If the retrieved slices do not cover the question, briefly explain what information is missing before offering general guidance. "
        "Do not hallucinate facts."
    ),
    SOURCE_KIND_HTS: (
        "You are a trade compliance assistant that consults the U.S. HTS CSV data. "
        "Use the retrieved chunks below to describe duty rates, classifications, and related guidance, citing each chunk number you reference (e.g., \"(Source 2)\"). "
        "When you cannot find a relevant chunk, explain the gap and avoid making up HTS numbers."
    ),
}

GENERATION_NO_CONTEXT_MESSAGE = (
    "The retriever did not return any snippets from {source_label}, so answer based only on the dataset's documented scope and be transparent that no direct hits were found."
)


def build_generation_messages(source_kind: str, source_label: str, context_text: str, question: str) -> list[dict]:
    system_prompt = GENERATION_SYSTEM_PROMPTS.get(source_kind, GENERATION_SYSTEM_PROMPTS[SOURCE_KIND_SDN])
    messages = [{"role": "system", "content": system_prompt}]
    if context_text:
        messages.append(
            {
                "role": "system",
                "content": f"Retrieved context from {source_label}:\n{context_text}",
            }
        )
    else:
        messages.append(
            {"role": "system", "content": GENERATION_NO_CONTEXT_MESSAGE.format(source_label=source_label)}
        )
    messages.append({"role": "user", "content": question})
    return messages


def main() -> None:
    st.set_page_config(page_title="Sanctions RAG Assistant", layout="wide")
    st.title("Sanctions RAG Assistant")
    st.caption("Uses OFAC SDN or HTS data as the knowledge source.")

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

    sidebar_sources: list[dict] = []
    sdn_path = Path("SDN_ENHANCED.ZIP")
    if sdn_path.exists():
        sidebar_sources.append(
            {
                "label": "OFAC SDN Enhanced (SDN_ENHANCED.ZIP)",
                "kind": SOURCE_KIND_SDN,
                "path": str(sdn_path),
            }
        )
    latest_hts = find_latest_hts_csv(Path("."))
    if latest_hts:
        sidebar_sources.append(
            {
                "label": f"US HTS {latest_hts.name}",
                "kind": SOURCE_KIND_HTS,
                "path": str(latest_hts),
            }
        )
    if not sidebar_sources:
        st.warning(
            "Drop SDN_ENHANCED.ZIP or an HTS CSV (e.g., hts_2026_revision_4_csv.csv) into the app root so the RAG index can be built."
        )
        return

    embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", DEFAULT_EMBEDDING_MODEL
    )

    try:
        source_labels = [source["label"] for source in sidebar_sources]
        default_index = 0
        with st.sidebar:
            st.header("Retrieval knobs")
            top_k = st.slider("Chunks to consider", 1, 5, DEFAULT_TOP_K)
            st.caption(
                "Searches the selected knowledge source for semantic matches, then passes the highest scoring chunks to the chat completion."
            )
            st.caption(
                "Temperature is fixed at 1.0 because this deployment currently only supports the default value."
            )
            if len(sidebar_sources) > 1:
                selected_label = st.radio("Knowledge source", source_labels, index=default_index)
            else:
                selected_label = source_labels[0]
            selected_source = sidebar_sources[source_labels.index(selected_label)]
        rag_index = load_rag_index(
            embedding_deployment,
            selected_source["kind"],
            selected_source["path"],
        )
    except Exception as exc:
        st.error(f"Failed to prepare the RAG index: {exc}")
        return

    with st.sidebar:
        st.caption("Current knowledge source: " + selected_label)

    question_prompt = QUESTION_PLACEHOLDERS.get(
        selected_source["kind"],
        "Ask about the selected knowledge source (provide enough detail for retrieval):",
    )
    question = st.text_area(
        question_prompt,
        height=140,
        key="prompt_input",
    )
    if st.button("Send query to Azure OpenAI", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before sending it to Azure OpenAI.")
            return

        reference_chunks = rag_index.search(question, top_k=top_k)
        if reference_chunks:
            context_text = _format_reference_sections(reference_chunks, selected_source["kind"])
            with st.expander(f"Validation snippets from {selected_label}", expanded=True):
                st.text_area(
                    "Top matches", value=context_text, height=240, max_chars=None, key="doc_view"
                )
        else:
            context_text = ""
            st.info("No semantically similar documents were found in the archive.")

        messages = build_generation_messages(
            selected_source["kind"],
            selected_label,
            context_text,
            question,
        )

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
