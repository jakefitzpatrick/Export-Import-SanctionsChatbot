"""Streamlit sandbox-aware chatbot that references multiple knowledge sources via RAG."""
import os
from pathlib import Path

import openai
import streamlit as st

from rag import (
    DEFAULT_EMBEDDING_MODEL,
    SOURCE_KIND_PDF,
    RagIndex,
    create_document_source,
)

DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50

# Smaller chunks with more overlap for richer context
CHUNK_CONFIG_BY_KIND = {
    SOURCE_KIND_PDF: {"chunk_size": 200, "chunk_overlap": 100},
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
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        chunk_index = item.get("chunk_index")
        chunk_number = chunk_index + 1 if chunk_index is not None else idx
        citation = f"(Source {source_label} | Source {chunk_number})"
        lines.append(
            f"{citation}\nScore: {item.get('score', 0):.3f}\n{item.get('chunk', '')}"
        )
    return "\n\n".join(lines)


QUESTION_PLACEHOLDER = "Ask about OFAC SDN Enhanced or U.S. HTS topics (provide enough detail for retrieval):"

COMBINED_SYSTEM_PROMPT = (
    'You are a compliance assistant that relies purely on the provided PDF. '
    'Always ground your answers in the retrieved evidence below, cite each chunk with its source label, chunk number, and page number (e.g., "(Source finalCopy_2026HTSRev4.pdf | Source 3 | Page 5)"). '
    'Prefer exact matches to the user query terms; if any necessary context is missing from the retrieved pages, explicitly flag which pages or sections are unavailable before offering general guidance. '
    'Do not hallucinate facts; begin every response with "Answer:" followed by a concise conclusion.'
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

    # locate finalCopy PDF (accept in project root or current working dir)
    pdf_filename = "finalCopy_2026HTSRev4.pdf"
    project_root = Path(__file__).parent.parent
    candidates = [project_root / pdf_filename, Path.cwd() / pdf_filename]
    pdf_path = next((p for p in candidates if p.exists()), None)
    if not pdf_path:
        st.error(f"{pdf_filename} not found; place it in {project_root} or the current folder.")
        return
    sidebar_sources = [{"label": pdf_path.name, "kind": SOURCE_KIND_PDF, "path": str(pdf_path)}]

    embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", DEFAULT_EMBEDDING_MODEL
    )

    try:
        source_labels = [source["label"] for source in sidebar_sources]
        with st.sidebar:
            st.header("Retrieval knobs")
            top_k = st.slider("Chunks to consider", 1, 10, DEFAULT_TOP_K)
            page_filter_input = st.text_input("Filter pages (comma-separated)", "")
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
        # parse page filters from input
        page_filters = {
            int(p.strip()) for p in page_filter_input.split(",") if p.strip().isdigit()
        }
        for source in sidebar_sources:
            label = source["label"]
            index = rag_indexes[label]
            # initial semantic retrieval (2× top_k)
            candidates = index.search(question, top_k=top_k * 2)
            # filter by page if requested
            if page_filters:
                candidates = [
                    c for c in candidates if c["metadata"].get("page") in page_filters
                ]
            # simple lexical rerank on query term hits
            terms = [t.lower() for t in question.split()]
            def lex_score(chunk: dict) -> int:
                text = chunk.get("chunk", "").lower()
                return sum(text.count(term) for term in terms)
            candidates.sort(key=lambda c: (lex_score(c), c.get("score", 0)), reverse=True)
            reference_chunks = candidates[:top_k]
            if reference_chunks:
                context_text = _format_reference_sections(reference_chunks, source["kind"], label)
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
