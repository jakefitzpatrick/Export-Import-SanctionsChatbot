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
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50
CHUNK_CONFIG_BY_KIND = {
    SOURCE_KIND_SDN: {"chunk_size": 220, "chunk_overlap": 60},
    SOURCE_KIND_HTS: {"chunk_size": 200, "chunk_overlap": 45},
}


@st.cache_resource(show_spinner=False)
def load_rag_index(
    embedding_model: str,
    source_kind: str,
    source_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RagIndex:
    document_source = create_document_source(source_kind, Path(source_path))
    index = RagIndex(
        document_source=document_source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    index.ensure_index(embedding_model)
    return index


def _format_reference_sections(results: list[dict], source_kind: str, source_label: str) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        metadata = item["metadata"] or {}
        chunk_index = item.get("chunk_index")
        chunk_number = chunk_index + 1 if chunk_index is not None else idx
        citation = f"(Source {source_label} | Source {chunk_number})"
        if source_kind == SOURCE_KIND_HTS:
            title = metadata.get("description") or metadata.get("hts_number") or "HTS chunk"
        else:
            title = metadata.get("entity_name") or metadata.get("primary_name") or "Unknown entity"
        identifier = metadata.get("entity_id") or metadata.get("row_number") or "N/A"
        lines.append(
            f"{citation}\n{title} (ID {identifier})\nScore: {item['score']:.3f}\n{item['chunk']}"
        )
    return "\n\n".join(lines)


QUESTION_PLACEHOLDER = "Ask about OFAC SDN Enhanced or U.S. HTS topics (provide enough detail for retrieval):"

COMBINED_SYSTEM_PROMPT = (
    'You are a compliance assistant that relies on both the OFAC SDN Enhanced List and the U.S. HTS CSV data. '
    'Always ground your answers in the retrieved evidence below and cite each chunk you reference with its source label and chunk number (e.g., "(Source OFAC SDN Enhanced | Source 1)"). '
    'If a relevant chunk is missing, explain the gap before offering general guidance, then answer the question. Do not hallucinate facts, and start every reply with "Answer:" followed by a concise conclusion.'
)

GENERATION_NO_CONTEXT_MESSAGE = (
    "The retriever did not return any snippets from any source, so be transparent that no direct hits were found and answer based on the documented scope of the datasets."
)

RESPONSE_STRUCTURE_INSTRUCTION = (
    "Structure your response in two parts: first a concise \"Answer:\" line that directly addresses the user question, then a \"Supporting Evidence:\" section with short bullet(s) referencing the chunk(s) you used. "
    "Each bullet must end with the citation in the format (Source <label> | Source <chunk number>) so users know exactly which chunk resolved the question."
)

def build_generation_messages(contexts: list[tuple[str, str]], question: str) -> list[dict]:
    messages = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    if contexts:
        for label, context_text in contexts:
            messages.append(
                {"role": "system", "content": f"Retrieved context from {label}:\n{context_text}"}
            )
    else:
        messages.append({"role": "system", "content": GENERATION_NO_CONTEXT_MESSAGE})
    messages.append({"role": "system", "content": RESPONSE_STRUCTURE_INSTRUCTION})
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
        with st.sidebar:
            st.header("Retrieval knobs")
            top_k = st.slider("Chunks to consider", 1, 5, DEFAULT_TOP_K)
            st.caption(
                "Searches the selected knowledge sources for semantic matches, then passes the highest scoring chunks to the chat completion."
            )
            st.caption(
                "Temperature is fixed at 1.0 because this deployment currently only supports the default value."
            )
            st.caption("Knowledge sources: " + ", ".join(source_labels))
        rag_indexes: dict[str, RagIndex] = {}
        for source in sidebar_sources:
            chunk_settings = CHUNK_CONFIG_BY_KIND.get(source["kind"], {})
            rag_indexes[source["label"]] = load_rag_index(
                embedding_deployment,
                source["kind"],
                source["path"],
                chunk_size=chunk_settings.get("chunk_size", DEFAULT_CHUNK_SIZE),
                chunk_overlap=chunk_settings.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            )
    except Exception as exc:
        st.error(f"Failed to prepare the RAG index: {exc}")
        return

    with st.sidebar:
        st.caption("Current knowledge sources: " + ", ".join(source_labels))

    question_prompt = QUESTION_PLACEHOLDER
    question = st.text_area(
        question_prompt,
        height=140,
        key="prompt_input",
    )
    if st.button("Send query to Azure OpenAI", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before sending it to Azure OpenAI.")
            return

        source_contexts: list[tuple[str, str]] = []
        previews: list[str] = []
        for source in sidebar_sources:
            label = source["label"]
            index = rag_indexes[label]
            reference_chunks = index.search(question, top_k=top_k)
            if reference_chunks:
                context_text = _format_reference_sections(
                    reference_chunks, source["kind"], label
                )
                source_contexts.append((label, context_text))
                previews.append(f"{label}:\n{context_text}")
        if source_contexts:
            with st.expander("Validation snippets from all sources", expanded=True):
                st.text_area(
                    "Top matches",
                    value="\n\n".join(previews),
                    height=260,
                    max_chars=None,
                    key="doc_view",
                )
        else:
            st.info("No semantically similar documents were found in the archive.")
        messages = build_generation_messages(source_contexts, question)

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
