from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Raw document before chunking."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval unit produced by Chunking."""

    chunk_id: str
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single item returned by Retrieval or Reranking."""

    source_id: str
    content: str
    score: float = 0.0
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output produced by Generation."""

    output: str
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    """Contextual information available to the Query component."""

    topic: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
