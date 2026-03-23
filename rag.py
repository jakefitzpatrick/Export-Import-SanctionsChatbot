"""RAG support: manage text chunks, embeddings and caching for multiple document sources."""
from __future__ import annotations

import csv
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import openai
import xml.etree.ElementTree as ET

CACHE_BASE = Path(".rag_cache")
CHUNKS_FILENAME = "chunks.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

HTS_FILE_PATTERN = "hts_*_csv.csv"
REVISION_PATTERN = re.compile(r"revision_(\d+)", re.IGNORECASE)

SOURCE_KIND_SDN = "sdn_enhanced"
SOURCE_KIND_HTS = "hts_csv"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _extract_revision_from_name(name: str) -> Optional[int]:
    match = REVISION_PATTERN.search(name)
    if match:
        return int(match.group(1))
    return None


def find_latest_hts_csv(root: Union[str, Path] = Path(".")) -> Optional[Path]:
    directory = Path(root)
    candidates = [p for p in directory.glob(HTS_FILE_PATTERN) if p.is_file()]
    if not candidates:
        return None

    def _score(path: Path) -> Tuple[int, float]:
        revision = _extract_revision_from_name(path.name) or 0
        return revision, path.stat().st_mtime

    return max(candidates, key=_score)


@dataclass
class DocumentRecord:
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentSource(ABC):
    @property
    @abstractmethod
    def cache_key(self) -> str:
        ...

    @property
    @abstractmethod
    def signature(self) -> dict:
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        ...

    @abstractmethod
    def iter_records(self) -> Iterable[DocumentRecord]:
        ...


def create_document_source(kind: str, path: Union[str, Path]) -> DocumentSource:
    if kind == SOURCE_KIND_SDN:
        return XmlZipDocumentSource(path)
    if kind == SOURCE_KIND_HTS:
        return CsvDocumentSource(path)
    raise ValueError(f"Unknown RAG source kind: {kind}")


class XmlZipDocumentSource(DocumentSource):
    def __init__(self, archive_path: Union[str, Path], xml_member: str = "SDN_ENHANCED.XML") -> None:
        self.archive_path = Path(archive_path)
        self.xml_member = xml_member
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

    @property
    def cache_key(self) -> str:
        return self.archive_path.stem

    @property
    def kind(self) -> str:
        return SOURCE_KIND_SDN

    @property
    def signature(self) -> dict:
        stat = self.archive_path.stat()
        return {
            "kind": self.kind,
            "path": str(self.archive_path),
            "xml_member": self.xml_member,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }

    def iter_records(self) -> Iterable[DocumentRecord]:
        with ZipFile(self.archive_path) as archive:
            try:
                with archive.open(self.xml_member) as xml_file:
                    context = ET.iterparse(xml_file, events=("start", "end"))
                    namespace: Optional[str] = None
                    for event, elem in context:
                        if namespace is None and event == "start":
                            if "}" in elem.tag:
                                namespace = elem.tag.split("}", 1)[0].strip("{}").strip()
                        if event == "end" and _strip_ns(elem.tag) == "entity":
                            metadata = self._entity_metadata(elem, namespace)
                            text = self._entity_text(elem, metadata)
                            yield DocumentRecord(text=text, metadata=metadata)
                            elem.clear()
            except KeyError as exc:
                raise ValueError(
                    f"{self.xml_member} not found inside {self.archive_path}"
                ) from exc

    def _entity_metadata(self, entity: ET.Element, namespace: Optional[str]) -> dict:
        entity_id = entity.get("id")
        primary_name = self._primary_name(entity, namespace)
        entity_type = self._text(entity, ["generalInfo", "entityType"], namespace)
        sanctions_list = ", ".join(
            [elem.text for elem in entity.findall(
                f"{{{namespace}}}sanctionsLists/{{{namespace}}}sanctionsList"
            ) if elem.text]
        )
        return {
            "entity_id": entity_id,
            "primary_name": primary_name,
            "entity_type": entity_type,
            "sanctions_list": sanctions_list,
        }

    def _entity_text(self, entity: ET.Element, metadata: dict) -> str:
        prefix = []
        if metadata.get("primary_name"):
            prefix.append(f"Name: {metadata['primary_name']}")
        if metadata.get("entity_id"):
            prefix.append(f"ID: {metadata['entity_id']}")
        if metadata.get("entity_type"):
            prefix.append(f"Type: {metadata['entity_type']}")
        if metadata.get("sanctions_list"):
            prefix.append(f"Lists: {metadata['sanctions_list']}")
        body = "\n".join(line.strip() for line in entity.itertext() if line.strip())
        return " | ".join(prefix) + "\n" + body if prefix else body

    def _text(self, element: ET.Element, path: List[str], namespace: Optional[str]) -> Optional[str]:
        target = element
        for part in path:
            tag = f"{{{namespace}}}{part}" if namespace else part
            target = target.find(tag)
            if target is None:
                return None
        return target.text

    def _primary_name(self, entity: ET.Element, namespace: Optional[str]) -> Optional[str]:
        name_tag = f"{{{namespace}}}name" if namespace else "name"
        translation_tag = f"{{{namespace}}}formattedFullName" if namespace else "formattedFullName"
        for name in entity.findall(name_tag):
            translated = name.find(translation_tag)
            if translated is not None and translated.text:
                return translated.text
        return None


class CsvDocumentSource(DocumentSource):
    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    @property
    def cache_key(self) -> str:
        return self.csv_path.stem

    @property
    def kind(self) -> str:
        return SOURCE_KIND_HTS

    @property
    def signature(self) -> dict:
        stat = self.csv_path.stat()
        return {
            "kind": self.kind,
            "path": str(self.csv_path),
            "revision": _extract_revision_from_name(self.csv_path.name),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }

    def iter_records(self) -> Iterable[DocumentRecord]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = self._row_to_text(row)
                if not text:
                    continue
                metadata = self._row_to_metadata(row)
                metadata["row_number"] = reader.line_num
                yield DocumentRecord(text=text, metadata=metadata)

    def _row_to_text(self, row: dict) -> str:
        def _clean(value: Optional[str]) -> str:
            return value.strip() if value else ""

        pieces: List[str] = []
        hts_number = _clean(row.get("HTS Number"))
        indent = _clean(row.get("Indent"))
        description = _clean(row.get("Description"))
        unit = _clean(row.get("Unit of Quantity"))
        general = _clean(row.get("General Rate of Duty"))
        special = _clean(row.get("Special Rate of Duty"))
        column2 = _clean(row.get("Column 2 Rate of Duty"))
        quota = _clean(row.get("Quota Quantity"))
        additional = _clean(row.get("Additional Duties"))

        if hts_number:
            pieces.append(f"HTS Number: {hts_number}")
        if indent:
            pieces.append(f"Indent: {indent}")
        if description:
            pieces.append(f"Description: {description}")
        if unit:
            pieces.append(f"Unit: {unit}")
        if general:
            pieces.append(f"General Rate: {general}")
        if special:
            pieces.append(f"Special Rate: {special}")
        if column2:
            pieces.append(f"Column 2 Rate: {column2}")
        if quota:
            pieces.append(f"Quota: {quota}")
        if additional:
            pieces.append(f"Additional Duties: {additional}")

        return " | ".join(pieces)

    def _row_to_metadata(self, row: dict) -> dict:
        metadata: dict = {}
        for key, value in row.items():
            cleaned = value.strip() if value else ""
            metadata[key.lower().replace(" ", "_")] = cleaned
        return metadata


class RagIndex:
    def __init__(
        self,
        document_source: Optional[DocumentSource] = None,
        cache_root: Union[str, Path] = CACHE_BASE,
        batch_size: int = 16,
    ) -> None:
        self.document_source = document_source or XmlZipDocumentSource("SDN_ENHANCED.ZIP")
        self.cache_dir = Path(cache_root) / self.document_source.cache_key
        self.chunks_path = self.cache_dir / CHUNKS_FILENAME
        self.embeddings_path = self.cache_dir / EMBEDDINGS_FILENAME
        self.manifest_path = self.cache_dir / MANIFEST_FILENAME
        self.batch_size = batch_size
        self.embedding_model: Optional[str] = None
        self._chunks: List[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._normalized_embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def ensure_index(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.embedding_model = embedding_model
        if self._is_cache_valid():
            self._load_cache()
        else:
            self._build_index()

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not query or not self._loaded:
            return []
        query_vector = self._embed_batch([query])[0]
        normalized_query = self._normalize_vectors(query_vector.reshape(1, -1))[0]
        assert self._normalized_embeddings is not None
        scores = np.dot(self._normalized_embeddings, normalized_query)
        if scores.size == 0:
            return []
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "chunk": self._chunks[idx]["text"],
                "score": float(scores[idx]),
                "metadata": self._chunks[idx].get("metadata", {}),
            }
            for idx in order
        ]

    def _is_cache_valid(self) -> bool:
        if not (self.manifest_path.exists() and self.chunks_path.exists() and self.embeddings_path.exists()):
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return (
            manifest.get("source_signature") == self.document_source.signature
            and manifest.get("embedding_model") == self.embedding_model
        )

    def _load_cache(self) -> None:
        self._chunks = [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._embeddings = np.load(self.embeddings_path)
        self._normalized_embeddings = self._normalize_vectors(self._embeddings)
        self._loaded = True

    def _build_index(self) -> None:
        records: List[dict] = []
        for record in self.document_source.iter_records():
            for idx, chunk_text in enumerate(self._chunk_text(record.text)):
                chunk_metadata = dict(record.metadata or {})
                chunk_metadata["chunk_index"] = idx
                records.append({"text": chunk_text, "metadata": chunk_metadata})
        if not records:
            raise RuntimeError("No document chunks found while building RAG index")
        embeddings = self._embed_batch([record["text"] for record in records])
        self._chunks = records
        self._embeddings = embeddings
        self._normalized_embeddings = self._normalize_vectors(embeddings)
        _ensure_directory(self.cache_dir)
        with open(self.chunks_path, "w", encoding="utf-8") as chunks_file:
            for record in records:
                json.dump(record, chunks_file, ensure_ascii=False)
                chunks_file.write("\n")
        np.save(self.embeddings_path, embeddings)
        self._write_manifest()
        self._loaded = True

    def _chunk_text(self, text: str, max_words: int = 250) -> Iterable[str]:
        if not text:
            return []
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i : i + max_words])

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model must be set before calling _embed_batch")
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = openai.embeddings.create(model=self.embedding_model, input=batch)
            items = getattr(response, "data", None) or response.get("data", [])
            embeddings.extend(
                np.array(
                    item.embedding if hasattr(item, "embedding") else item["embedding"],
                    dtype=np.float32,
                )
                for item in items
            )
        return np.vstack(embeddings)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return vectors / norms

    def _write_manifest(self) -> None:
        manifest = {
            "source_signature": self.document_source.signature,
            "embedding_model": self.embedding_model,
            "chunk_count": len(self._chunks),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
