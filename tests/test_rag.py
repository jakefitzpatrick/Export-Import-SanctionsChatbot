import typing

import numpy as np
import pytest

from rag import DocumentRecord, DocumentSource, RagIndex


class SimpleDocumentSource(DocumentSource):
    def __init__(self, records: typing.Sequence[DocumentRecord]) -> None:
        self._records = list(records)

    @property
    def cache_key(self) -> str:
        return "simple-source"

    @property
    def signature(self) -> dict:
        return {
            "kind": "simple",
            "record_count": len(self._records),
        }

    @property
    def kind(self) -> str:
        return "simple"

    def iter_records(self):
        yield from self._records


def test_search_prefers_domain_chunk(tmp_path):
    records = [
        DocumentRecord(text="digital assets overview", metadata={"entity_id": "target"}),
        DocumentRecord(text="other structured info", metadata={"entity_id": "aux"}),
        DocumentRecord(text="ancillary data", metadata={"entity_id": "filler"}),
    ]
    source = SimpleDocumentSource(records)
    index = RagIndex(document_source=source, cache_root=tmp_path)

    vector_map = {
        "digital assets overview": [1.0, 0.0, 0.0],
        "other structured info": [0.0, 1.0, 0.0],
        "ancillary data": [0.0, 0.0, 1.0],
        "digital assets": [1.0, 0.0, 0.0],
    }

    def fake_embed(texts):
        vectors = []
        for text in texts:
            value = vector_map.get(text, [0.1, 0.1, 0.1])
            vectors.append(np.array(value, dtype=np.float32))
        return np.vstack(vectors)

    index._embed_batch = fake_embed
    index.ensure_index("test-model")

    results = index.search("digital assets", top_k=3)
    assert results, "search should return at least one chunk"
    assert results[0]["metadata"]["entity_id"] == "target"
    assert len(results) == 3
    assert all(isinstance(r["score"], float) for r in results)
