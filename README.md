# SemaCache Python SDK

Official Python client for [semacache.io](https://semacache.io) — semantic caching for LLM APIs.

Zero dependencies on OpenAI. Just `httpx` under the hood.

## Installation

```bash
pip install semacache
```

## Quick Start

```python
from semacache import SemaCache

client = SemaCache(api_key="sc-your-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is semantic caching?"}],
)
print(response.choices[0].message.content)

# Cache metadata is always available
print(response.cache.match_type)   # "EXACT", "SEMANTIC", or None
print(response.cache.confidence)   # 0.991 (for semantic hits)
```

## Configuration

```python
client = SemaCache(
    api_key="sc-your-key",
    upstream_api_key="sk-your-openai-key",   # optional: pass inline instead of dashboard
    similarity_threshold=0.90,                # optional: override default (0.95)
    cache_ttl=3600,                           # optional: cache for 1 hour
)
```

## Per-Request Overrides

```python
response = client.chat.completions.create(
    model="grok-3-mini",
    messages=[{"role": "user", "content": "Hello"}],
    similarity_threshold=0.85,
    cache_ttl=7200,
    no_cache=True,   # skip cache read, still store
)
```

## Image Generation

```python
image = client.images.generate(
    prompt="A sunset over mountains",
    model="gpt-image-1",     # or "imagen-4.0-generate-001", "grok-imagine-image"
    size="1024x1024",
)
print(image.data[0].url)
print(image.cache.match_type)
```

## Video Generation

```python
video = client.videos.generate(
    prompt="A drone flyover of a city at sunset",
    model="veo-3.0-generate-001",   # or "veo-3.1-generate-preview", "grok-imagine-video"
    duration_seconds=8,
    aspect_ratio="16:9",
)
print(video.data[0].url)
print(video.cache.match_type)
```

## Passthrough params

Any keyword arg beyond the SDK's named ones is forwarded to the upstream
provider verbatim. New OpenAI / Gemini / xAI params work the moment the
provider ships them — you don't need to wait for the SDK to catch up. Use
`extra_body` for provider-specific extensions.

```python
# Chat — forward temperature, tools, reasoning_effort, response_format, …
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Summarize SemCache in one line"}],
    temperature=0.2,
    reasoning_effort="high",
    response_format={"type": "json_object"},
)

# Image — forward style, seed, negative_prompt, extra_body, …
image = client.images.generate(
    prompt="A red square on a white background",
    model="imagen-4.0-generate-001",
    seed=42,
    negative_prompt="blurry, low quality",
    aspect_ratio="16:9",      # Imagen-native
)

# Video — forward resolution, enhance_prompt, seed, negative_prompt, …
video = client.videos.generate(
    prompt="A drone flyover of a city",
    model="veo-3.0-generate-001",
    resolution="1080p",
    enhance_prompt=True,
    negative_prompt="blurry",
)

# Gemini-specific escape hatch
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]},
)
```

## Async

```python
from semacache import SemaCacheAsync

async with SemaCacheAsync(api_key="sc-your-key") as client:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
```

## Supported Models

All models supported by semacache.io work through this SDK:

**Chat**
- **OpenAI**: gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o3, o3-mini, o4-mini
- **Gemini**: gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-3.1-flash-lite-preview, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- **xAI**: grok-4.20, grok-4, grok-4-fast, grok-3, grok-3-mini, grok-3-fast

**Images**
- **OpenAI**: gpt-image-1.5, gpt-image-1, gpt-image-1-mini, dall-e-3, dall-e-2
- **Google Imagen**: imagen-4.0-generate-001, imagen-4.0-ultra-generate-001, imagen-4.0-fast-generate-001
- **xAI**: grok-imagine-image, grok-imagine-image-pro

**Videos**
- **Google Veo**: veo-3.1-generate-preview, veo-3.1-fast-generate-preview, veo-3.1-lite-generate-preview, veo-3.0-generate-001, veo-3.0-fast-generate-001, veo-2.0-generate-001
- **xAI**: grok-imagine-video

**Custom**: any OpenAI-compatible endpoint registered in the dashboard

## Links

- [semacache.io](https://semacache.io)
- [Documentation](https://semacache.io/docs)
- [Dashboard](https://semacache.io/dashboard)
