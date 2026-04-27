"""Shared WTB response-cache helpers for swappable RAG components.

The helpers in this module keep cache wiring at the canonical component
boundary.  LongRAG-style readers, STORM-style writers, Self-RAG shims, and
common OminiRAG components can all share the same persistent WTB cache when
they use an LLM object exposing ``complete(system, user, **kwargs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import os
from pathlib import Path
from typing import Any


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class WTBCacheConfig:
    """Normalized WTB cache configuration shared by all swappable systems."""

    enabled: bool = True
    cache_path: str | None = None
    api_key: str = ""
    base_url: str | None = "https://api.openai.com/v1"
    text_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    debug: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        cache_path: str | None = None,
        enabled: bool | None = None,
    ) -> "WTBCacheConfig":
        """Build config from the standard WTB/OminiRAG environment."""
        if enabled is None:
            enabled = _as_bool(
                os.getenv("RAG_USE_WTB_CACHE"),
                _as_bool(os.getenv("WTB_LLM_RESPONSE_CACHE_ENABLED"), True),
            )

        resolved_cache_path = cache_path or os.getenv("WTB_LLM_CACHE_PATH")
        workspace_dir = os.getenv("RAG_WORKSPACE_DIR")
        if not resolved_cache_path and workspace_dir:
            resolved_cache_path = str(
                Path(workspace_dir) / "wtb_data" / "llm_response_cache.db"
            )

        api_key = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", ""))
        base_url = os.getenv(
            "OPENAI_BASE_URL",
            os.getenv("LLM_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")),
        )
        text_model = os.getenv("WTB_LLM_TEXT_MODEL", os.getenv("DEFAULT_LLM", "gpt-4o-mini"))
        embedding_model = os.getenv(
            "WTB_LLM_EMBEDDING_MODEL",
            os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )

        return cls(
            enabled=bool(enabled),
            cache_path=resolved_cache_path,
            api_key=api_key,
            base_url=base_url or None,
            text_model=text_model,
            embedding_model=embedding_model,
            debug=_as_bool(os.getenv("WTB_LLM_DEBUG"), False),
        )

    @property
    def cache_active(self) -> bool:
        """True when persistent WTB response caching is configured."""
        return self.enabled and bool(self.cache_path)

    def as_env(self) -> dict[str, str]:
        """Return the standardized environment for subprocess execution."""
        env = {
            "RAG_USE_WTB_CACHE": "true" if self.enabled else "false",
            "WTB_LLM_RESPONSE_CACHE_ENABLED": "true" if self.enabled else "false",
            "WTB_LLM_TEXT_MODEL": self.text_model,
            "WTB_LLM_EMBEDDING_MODEL": self.embedding_model,
        }
        if self.cache_path:
            env["WTB_LLM_CACHE_PATH"] = self.cache_path
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            env["OPENAI_BASE_URL"] = self.base_url
        return env


class WTBCachedLLM:
    """LLM facade backed by WTB's persistent text-generation cache."""

    def __init__(
        self,
        config: WTBCacheConfig | None = None,
        *,
        system_name: str = "ominirag",
        node_path: str = "llm",
        model: str | None = None,
    ) -> None:
        self.config = config or WTBCacheConfig.from_env()
        self.system_name = system_name
        self.node_path = node_path
        self.model = model or self.config.text_model
        self._service: Any | None = None
        self._last_result: Any | None = None

    @property
    def service(self) -> Any:
        """Return the lazily constructed WTB LLM service."""
        if self._service is None:
            try:
                from wtb.infrastructure.llm import LangChainOpenAIConfig, get_service
            except ImportError as exc:
                raise ImportError(
                    "WTB cache support requires a WTB build with "
                    "wtb.infrastructure.llm. Install the WTB cache wheel first."
                ) from exc

            service_config = LangChainOpenAIConfig(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                default_text_model=self.config.text_model,
                default_embedding_model=self.config.embedding_model,
                response_cache_path=self.config.cache_path,
                response_cache_enabled=self.config.enabled,
                debug=self.config.debug,
            )
            self._service = get_service(service_config)
        return self._service

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Generate text through WTB, returning cached text on repeated calls."""
        result = self.service.generate_text_result(
            prompt=user,
            model=kwargs.get("model", self.model),
            system_prompt=system,
            temperature=float(kwargs.get("temperature", 0.0)),
            max_tokens=kwargs.get("max_tokens"),
        )
        self._last_result = result
        return result.text

    def cache_stats(self) -> dict[str, Any]:
        return dict(self.service.get_cache_stats())

    def wtb_cache_metadata(self) -> dict[str, Any]:
        """Return cache metadata suitable for GenerationResult.metadata."""
        stats = self.cache_stats()
        metadata: dict[str, Any] = {
            "enabled": bool(stats.get("enabled")),
            "path": stats.get("path"),
            "entries": stats.get("entries", 0),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "writes": stats.get("writes", 0),
            "system": self.system_name,
            "node_path": self.node_path,
        }
        if self._last_result is not None:
            metadata.update(
                {
                    "last_cache_hit": bool(self._last_result.cache_hit),
                    "last_cache_key": self._last_result.cache_key,
                    "last_model": self._last_result.model,
                    "last_duration_ms": self._last_result.duration_ms,
                }
            )
        return metadata


def get_wtb_cache_metadata(source: Any) -> dict[str, Any] | None:
    """Extract WTB cache metadata from an object or one of its LLM/model attrs."""
    if source is None:
        return None

    for method_name in ("wtb_cache_metadata", "cache_metadata"):
        method = getattr(source, method_name, None)
        if callable(method):
            metadata = method()
            if metadata:
                return dict(metadata)

    for attr_name in ("llm", "model"):
        nested = getattr(source, attr_name, None)
        if nested is not None and nested is not source:
            metadata = get_wtb_cache_metadata(nested)
            if metadata:
                return metadata

    return None


def attach_wtb_cache_metadata(
    metadata: dict[str, Any] | None,
    source: Any,
    *,
    key: str = "wtb_cache",
) -> dict[str, Any]:
    """Copy metadata and add a ``wtb_cache`` block when available."""
    merged = dict(metadata or {})
    cache_metadata = get_wtb_cache_metadata(source)
    if cache_metadata:
        merged[key] = cache_metadata
    return merged


@dataclass(frozen=True)
class WTBSystemCacheStatus:
    """Import/status report for cache wiring on an optional swappable system."""

    name: str
    import_path: str
    importable: bool
    cache_surface: str
    detail: str


_SYSTEM_IMPORTS = {
    "longrag": "longRAG_example.longrag_langgraph.main_pipeline",
    "storm": "storm.storm_langgraph",
    "selfrag": "selfrag.modular_pipeline",
}


def inspect_swappable_system_cache_support() -> list[WTBSystemCacheStatus]:
    """Report which optional system packages are present for cache wiring."""
    statuses: list[WTBSystemCacheStatus] = []
    for name, import_path in _SYSTEM_IMPORTS.items():
        try:
            importable = find_spec(import_path) is not None
        except (ImportError, ModuleNotFoundError):
            importable = False
        if importable:
            detail = "importable; use WTBCachedLLM or cached model shim in component builders"
        else:
            detail = (
                "not importable in this checkout; the gitlink directory is present "
                "but its package contents are unavailable"
            )
        statuses.append(
            WTBSystemCacheStatus(
                name=name,
                import_path=import_path,
                importable=importable,
                cache_surface="rag_contracts.wtb_cache.WTBCachedLLM",
                detail=detail,
            )
        )
    return statuses
