"""SemaCache Python SDK — standalone HTTP client for semacache.io."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://www.semacache.io/api/v1"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CacheInfo:
    """SemaCache-specific metadata from response headers."""
    match_type: str | None = None   # EXACT, SEMANTIC, or None (miss)
    confidence: float | None = None
    latency_ms: float | None = None


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: list[Choice]
    usage: Usage
    cache: CacheInfo
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_response(cls, data: dict[str, Any], headers: httpx.Headers) -> "ChatCompletion":
        choices = [
            Choice(
                index=c.get("index", i),
                message=Message(
                    role=c["message"]["role"],
                    content=c["message"].get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            )
            for i, c in enumerate(data.get("choices", []))
        ]
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        conf = headers.get("x-semcache-confidence")
        lat = headers.get("x-semcache-latency-ms")
        cache = CacheInfo(
            match_type=headers.get("x-semcache-match-type"),
            confidence=float(conf) if conf else None,
            latency_ms=float(lat) if lat else None,
        )
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            cache=cache,
            raw=data,
        )


def _parse_cache_info(headers: httpx.Headers) -> CacheInfo:
    conf = headers.get("x-semcache-confidence")
    lat = headers.get("x-semcache-latency-ms")
    return CacheInfo(
        match_type=headers.get("x-semcache-match-type"),
        confidence=float(conf) if conf else None,
        latency_ms=float(lat) if lat else None,
    )


def _build_cache_headers(
    similarity_threshold: float | None,
    cache_ttl: int | None,
    no_cache: bool,
    no_store: bool,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if similarity_threshold is not None:
        headers["x-similarity-threshold"] = str(similarity_threshold)
    if cache_ttl is not None:
        headers["x-cache-ttl"] = str(cache_ttl)
    if no_cache:
        headers["Cache-Control"] = "no-cache"
    elif no_store:
        headers["Cache-Control"] = "no-store"
    return headers


@dataclass
class ImageData:
    url: str


@dataclass
class ImageGeneration:
    data: list[ImageData]
    cache: CacheInfo
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_response(cls, data: dict[str, Any], headers: httpx.Headers) -> "ImageGeneration":
        items = [ImageData(url=d.get("url", "")) for d in data.get("data", [])]
        return cls(data=items, cache=_parse_cache_info(headers), raw=data)


@dataclass
class VideoData:
    url: str


@dataclass
class VideoGeneration:
    data: list[VideoData]
    cache: CacheInfo
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_response(cls, data: dict[str, Any], headers: httpx.Headers) -> "VideoGeneration":
        items = [VideoData(url=d.get("url", "")) for d in data.get("data", [])]
        return cls(data=items, cache=_parse_cache_info(headers), raw=data)


class SemaCache:
    """Synchronous SemaCache client.

    Usage::

        from semacache import SemaCache

        client = SemaCache(api_key="sc-your-key")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)
        print(response.cache.match_type)  # EXACT, SEMANTIC, or None
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        upstream_api_key: str | None = None,
        similarity_threshold: float | None = None,
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if upstream_api_key:
            headers["x-upstream-api-key"] = upstream_api_key
        if similarity_threshold is not None:
            headers["x-similarity-threshold"] = str(similarity_threshold)
        if cache_ttl is not None:
            headers["x-cache-ttl"] = str(cache_ttl)

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
        )
        self.chat = self._ChatNamespace(self)
        self.images = self._ImagesNamespace(self)
        self.videos = self._VideosNamespace(self)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SemaCache":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    class _ChatNamespace:
        def __init__(self, client: "SemaCache") -> None:
            self.completions = SemaCache._CompletionsNamespace(client)

    class _CompletionsNamespace:
        def __init__(self, client: "SemaCache") -> None:
            self._client = client

        def create(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> ChatCompletion:
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
            resp = self._client._client.post("/chat/completions", json=body, headers=h)
            resp.raise_for_status()
            return ChatCompletion.from_response(resp.json(), resp.headers)

    class _ImagesNamespace:
        def __init__(self, client: "SemaCache") -> None:
            self._client = client

        def generate(
            self,
            *,
            prompt: str,
            model: str = "gpt-image-1",
            n: int = 1,
            size: str = "1024x1024",
            quality: str = "standard",
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> ImageGeneration:
            """Generate an image. Any extra kwargs (``style``, ``response_format``,
            ``seed``, ``negative_prompt``, ``extra_body``…) are forwarded to the
            upstream provider verbatim."""
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {
                "prompt": prompt, "model": model, "n": n, "size": size, "quality": quality,
                **kwargs,
            }
            resp = self._client._client.post("/images/generations", json=body, headers=h)
            resp.raise_for_status()
            return ImageGeneration.from_response(resp.json(), resp.headers)

    class _VideosNamespace:
        def __init__(self, client: "SemaCache") -> None:
            self._client = client

        def generate(
            self,
            *,
            prompt: str,
            model: str = "veo-2.0-generate-001",
            duration_seconds: int = 8,
            aspect_ratio: str = "16:9",
            n: int = 1,
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> VideoGeneration:
            """Generate a video. Any extra kwargs (``negative_prompt``, ``seed``,
            ``resolution``, ``enhance_prompt``, ``extra_body``…) are forwarded
            to the upstream provider verbatim."""
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {
                "prompt": prompt, "model": model, "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio, "n": n,
                **kwargs,
            }
            resp = self._client._client.post("/videos/generations", json=body, headers=h)
            resp.raise_for_status()
            return VideoGeneration.from_response(resp.json(), resp.headers)


class SemaCacheAsync:
    """Async SemaCache client.

    Usage::

        from semacache import SemaCacheAsync

        client = SemaCacheAsync(api_key="sc-your-key")

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        upstream_api_key: str | None = None,
        similarity_threshold: float | None = None,
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if upstream_api_key:
            headers["x-upstream-api-key"] = upstream_api_key
        if similarity_threshold is not None:
            headers["x-similarity-threshold"] = str(similarity_threshold)
        if cache_ttl is not None:
            headers["x-cache-ttl"] = str(cache_ttl)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
        )
        self.chat = self._ChatNamespace(self)
        self.images = self._ImagesNamespace(self)
        self.videos = self._VideosNamespace(self)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SemaCacheAsync":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    class _ChatNamespace:
        def __init__(self, client: "SemaCacheAsync") -> None:
            self.completions = SemaCacheAsync._CompletionsNamespace(client)

    class _CompletionsNamespace:
        def __init__(self, client: "SemaCacheAsync") -> None:
            self._client = client

        async def create(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> ChatCompletion:
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
            resp = await self._client._client.post("/chat/completions", json=body, headers=h)
            resp.raise_for_status()
            return ChatCompletion.from_response(resp.json(), resp.headers)

    class _ImagesNamespace:
        def __init__(self, client: "SemaCacheAsync") -> None:
            self._client = client

        async def generate(
            self,
            *,
            prompt: str,
            model: str = "gpt-image-1",
            n: int = 1,
            size: str = "1024x1024",
            quality: str = "standard",
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> ImageGeneration:
            """Generate an image. Any extra kwargs (``style``, ``response_format``,
            ``seed``, ``negative_prompt``, ``extra_body``…) are forwarded to the
            upstream provider verbatim."""
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {
                "prompt": prompt, "model": model, "n": n, "size": size, "quality": quality,
                **kwargs,
            }
            resp = await self._client._client.post("/images/generations", json=body, headers=h)
            resp.raise_for_status()
            return ImageGeneration.from_response(resp.json(), resp.headers)

    class _VideosNamespace:
        def __init__(self, client: "SemaCacheAsync") -> None:
            self._client = client

        async def generate(
            self,
            *,
            prompt: str,
            model: str = "veo-2.0-generate-001",
            duration_seconds: int = 8,
            aspect_ratio: str = "16:9",
            n: int = 1,
            similarity_threshold: float | None = None,
            cache_ttl: int | None = None,
            no_cache: bool = False,
            no_store: bool = False,
            **kwargs: Any,
        ) -> VideoGeneration:
            """Generate a video. Any extra kwargs (``negative_prompt``, ``seed``,
            ``resolution``, ``enhance_prompt``, ``extra_body``…) are forwarded
            to the upstream provider verbatim."""
            h = _build_cache_headers(similarity_threshold, cache_ttl, no_cache, no_store)
            body: dict[str, Any] = {
                "prompt": prompt, "model": model, "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio, "n": n,
                **kwargs,
            }
            resp = await self._client._client.post("/videos/generations", json=body, headers=h)
            resp.raise_for_status()
            return VideoGeneration.from_response(resp.json(), resp.headers)
