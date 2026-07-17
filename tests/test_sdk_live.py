"""Live smoke tests for the SemaCache Python SDK.

Exercises the real SDK against a live Cloud Run deployment and real upstream
providers. Verifies that arbitrary non-core parameters (``temperature``,
``max_completion_tokens``, ``response_format``, ``seed``, ``negative_prompt``,
…) are transparently forwarded through SemCache to the upstream provider.

Setup
-----
Reads configuration from ``apps/cache-service/.env.local``:

- ``DASHBOARD_SECRET`` (required) — used to mint a per-run test API key.
- ``OPENAI_API_KEY``   (required for chat + OpenAI image tests).
- ``GEMINI_API_KEY``   (required for ``--include-image=gemini``,
                       ``--include-video=veo``).
- ``XAI_API_KEY``      (required for xAI image / video tests).
- ``SEMCACHE_URL``     (optional, defaults to the prod Cloud Run URL).
- ``SEMCACHE_API_KEY`` (optional override — if set, skips the minting step).

Each run mints a fresh API key under a unique ``x-user-id`` so per-user key
limits don't interfere with repeated runs. The key is deleted in a
``finally`` block.

Usage
-----
From repo root::

    # cheap: chat only (one OpenAI call, ~$0.0001)
    python dev/python/tests/test_sdk_live.py

    # add image coverage (one OpenAI image call, ~$0.04)
    python dev/python/tests/test_sdk_live.py --include-image

    # prove the params_hash semantic-tier filter actually excludes
    # mismatched extras (4 image generations, ~$0.05)
    python dev/python/tests/test_sdk_live.py --include-params-isolation

    # add video coverage (SLOW — minutes; Veo / xAI Grok)
    python dev/python/tests/test_sdk_live.py --include-video

    # run everything including the async client
    python dev/python/tests/test_sdk_live.py --all

Or via pytest::

    pytest dev/python/tests/test_sdk_live.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Make the sibling ``semacache`` package importable when running as a script.
_THIS = Path(__file__).resolve()
_SDK_ROOT = _THIS.parents[1]
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))

from semacache import SemaCache, SemaCacheAsync  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────

_DEFAULT_CLOUD_RUN = "https://semcache-cache-service-619993309587.us-central1.run.app"
_DEFAULT_PROXY = "https://www.semacache.io/api"

# Walk up to the repo root and pull env from apps/cache-service/.env.local so
# the script works without a shell-level ``source`` step.
_REPO_ROOT = _THIS.parents[3]
_ENV_FILE = _REPO_ROOT / "apps" / "cache-service" / ".env.local"
if _ENV_FILE.is_file():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

SEMCACHE_URL = os.environ.get("SEMCACHE_URL", _DEFAULT_CLOUD_RUN).rstrip("/")
DASHBOARD_SECRET = os.environ.get("DASHBOARD_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# A per-run user id keeps the per-user API-key limit from blocking re-runs.
TEST_USER_ID = f"sdk-live-smoke-{int(time.time())}"


# ── Output helpers ────────────────────────────────────────────────────────


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _dim(s: str) -> str:
    return f"\033[90m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ── API-key bootstrap ─────────────────────────────────────────────────────


def _mint_api_key() -> tuple[str, str | None]:
    """Mint a fresh SemaCache API key for this run.

    Returns ``(raw_key, key_id)``. If ``SEMCACHE_API_KEY`` is set in the
    environment, that value is used and no server-side key is created
    (``key_id`` is ``None`` in that case).
    """
    override = os.environ.get("SEMCACHE_API_KEY")
    if override:
        print(_dim(f"  using SEMCACHE_API_KEY from env (preview: {override[:10]}…)"))
        return override, None

    if not DASHBOARD_SECRET:
        raise SystemExit(
            "DASHBOARD_SECRET missing — cannot mint a test key. "
            "Set it in apps/cache-service/.env.local or the environment."
        )

    name = f"sdk-live-smoke-{int(time.time())}"
    r = httpx.post(
        f"{SEMCACHE_URL}/dashboard/api-keys",
        headers={
            "x-dashboard-secret": DASHBOARD_SECRET,
            "x-user-id": TEST_USER_ID,
            "Content-Type": "application/json",
        },
        json={"name": name},
        timeout=30,
    )
    if r.status_code != 200:
        raise SystemExit(f"Failed to mint API key: {r.status_code} {r.text[:300]}")
    data = r.json()
    raw = data.get("raw_key") or data.get("key") or data.get("api_key")
    key_id = data.get("id")
    if not raw:
        raise SystemExit(f"Mint response missing raw key: {data}")
    print(_dim(f"  minted test key id={key_id} name={name}"))
    return raw, key_id


def _delete_api_key(key_id: str | None) -> None:
    if not key_id or not DASHBOARD_SECRET:
        return
    try:
        r = httpx.delete(
            f"{SEMCACHE_URL}/dashboard/api-keys/{key_id}",
            headers={
                "x-dashboard-secret": DASHBOARD_SECRET,
                "x-user-id": TEST_USER_ID,
            },
            timeout=15,
        )
        if r.status_code in (200, 204):
            print(_dim(f"  deleted test key id={key_id}"))
        else:
            print(_dim(f"  WARN: failed to delete key {key_id}: {r.status_code}"))
    except httpx.HTTPError as e:
        print(_dim(f"  WARN: failed to delete key {key_id}: {e}"))


# ── Test cases ────────────────────────────────────────────────────────────


def _check(label: str, cond: bool, detail: str = "") -> bool:
    mark = _green("  ✓") if cond else _red("  ✗")
    extra = f"  {_dim(detail)}" if detail else ""
    print(f"{mark} {label}{extra}")
    return cond


def _safe(fn: Any) -> bool:
    """Run a test fn, converting any exception into a visible failure so one
    broken test doesn't short-circuit the whole run."""
    try:
        return bool(fn())
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:  # noqa: BLE001
            pass
        print(_red(f"  ✗ HTTP {e.response.status_code}: {body}"))
        return False
    except Exception as e:  # noqa: BLE001
        print(_red(f"  ✗ exception: {type(e).__name__}: {e}"))
        return False


def test_chat_passthrough(client: SemaCache) -> bool:
    """OpenAI chat with non-core params that went through ``extra_body`` in the
    old whitelist world. If passthrough works end-to-end these are accepted
    by OpenAI and we get a coherent response back."""
    print(_bold("\n[chat] OpenAI gpt-4o-mini with temperature + max_completion_tokens + response_format"))
    if not OPENAI_API_KEY:
        print(_dim("  skipped: OPENAI_API_KEY not set"))
        return True
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with exactly the word: ok"}],
        temperature=0.1,
        max_completion_tokens=5,
        response_format={"type": "text"},
        no_cache=True,
        no_store=True,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    ok = True
    ok &= _check("got a response", bool(text), f"got: {text!r}")
    ok &= _check(
        "response mentions 'ok'",
        "ok" in text,
        f"content: {text!r}",
    )
    ok &= _check(
        "cache metadata returned",
        resp.cache.match_type is not None,
        f"match_type={resp.cache.match_type} latency_ms={resp.cache.latency_ms}",
    )
    ok &= _check(
        "usage tokens populated",
        resp.usage.total_tokens > 0,
        f"total_tokens={resp.usage.total_tokens}",
    )
    return ok


def test_chat_extra_body(client: SemaCache) -> bool:
    """extra_body is the explicit escape hatch — anything nested there must
    be forwarded to the upstream provider without being stripped."""
    print(_bold("\n[chat] extra_body escape hatch (OpenAI gpt-4o-mini)"))
    if not OPENAI_API_KEY:
        print(_dim("  skipped: OPENAI_API_KEY not set"))
        return True
    # OpenAI silently ignores unknown keys in extra_body, so this mainly
    # proves the SDK didn't drop the key on its own. We use a real-but-
    # optional param (``user``) to prove it still makes it to upstream.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say: yes"}],
        max_completion_tokens=3,
        extra_body={"user": "sdk-live-smoke"},
        no_cache=True,
        no_store=True,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    return _check("got response with extra_body attached", bool(text), f"got: {text!r}")


def test_image_passthrough(client: SemaCache) -> bool:
    """OpenAI image generation with a passthrough param (``user``) that isn't
    in the SDK's named signature. If passthrough works the proxy forwards it
    to OpenAI Images and we get a URL back."""
    print(_bold("\n[image] OpenAI gpt-image-1 with passthrough params"))
    if not OPENAI_API_KEY:
        print(_dim("  skipped: OPENAI_API_KEY not set"))
        return True
    img = client.images.generate(
        prompt="A tiny red cube on a plain white background, minimalist",
        model="gpt-image-1",
        n=1,
        size="1024x1024",
        quality="low",
        user="sdk-live-smoke",
        no_cache=True,
        no_store=True,
    )
    ok = True
    ok &= _check("got image data", bool(img.data), f"count={len(img.data)}")
    ok &= _check(
        "image url is an http(s) URL",
        bool(img.data) and img.data[0].url.startswith("http"),
        img.data[0].url if img.data else "<none>",
    )
    ok &= _check(
        "cache metadata returned",
        img.cache.match_type is not None,
        f"match_type={img.cache.match_type}",
    )
    return ok


def test_image_params_isolation(client: SemaCache) -> bool:
    """Prove the ``params_hash`` filter on the image semantic tier works.

    Same prompt, varying the ``user`` passthrough param. Order matters
    because the test asserts the cache state evolves correctly:

      1. user="iso-A"   → NATIVE  (cold tenant, nothing cached)
      2. user="iso-A"   → EXACT   (Redis hit on identical request)
      3. user="iso-B"   → NATIVE  (different params_hash → call-1 entry
                                   is filtered out of the semantic match)
      4. (no user)      → NATIVE  (NULL params_hash → entries from
                                   calls 1 & 3 are filtered out too)

    Before the fix, calls 3 and 4 would have been SEMANTIC hits returning
    the image generated under ``user="iso-A"`` — the silent-wrongness
    bug. After the fix, ``params_hash`` filters them out and we
    correctly fall through to a fresh upstream generation.

    Costs ~$0.05 (four ``gpt-image-1`` low-quality calls).
    """
    print(_bold("\n[image] params_hash isolation (4 paid generations, ~$0.05)"))
    if not OPENAI_API_KEY:
        print(_dim("  skipped: OPENAI_API_KEY not set"))
        return True

    # Unique-per-run prompt so prior runs of this test (with the same
    # SemCache user_id, if SEMCACHE_API_KEY is overridden) can't pollute
    # the cache and pre-empt the expected NATIVE/EXACT pattern.
    prompt = (
        f"a single tiny black square centered on a plain white background, "
        f"marker {int(time.time())}"
    )
    common: dict[str, Any] = dict(
        prompt=prompt,
        model="gpt-image-1",
        n=1,
        size="1024x1024",
        quality="low",
    )

    img1 = client.images.generate(**common, user="iso-A")
    ok = _check(
        "call 1 (user=iso-A) → NATIVE (cold)",
        img1.cache.match_type == "NATIVE",
        f"got match_type={img1.cache.match_type}",
    )

    img2 = client.images.generate(**common, user="iso-A")
    ok &= _check(
        "call 2 (user=iso-A repeat) → EXACT",
        img2.cache.match_type == "EXACT",
        f"got match_type={img2.cache.match_type}",
    )

    img3 = client.images.generate(**common, user="iso-B")
    ok &= _check(
        "call 3 (user=iso-B) → NATIVE  (params_hash filter excludes call-1 entry; was SEMANTIC pre-fix)",
        img3.cache.match_type == "NATIVE",
        f"got match_type={img3.cache.match_type}",
    )

    img4 = client.images.generate(**common)
    ok &= _check(
        "call 4 (no user)   → NATIVE  (NULL params_hash filter excludes calls 1 & 3; was SEMANTIC pre-fix)",
        img4.cache.match_type == "NATIVE",
        f"got match_type={img4.cache.match_type}",
    )

    return ok


def test_video_passthrough(client: SemaCache) -> bool:
    """xAI grok-imagine-video with passthrough params. Very slow — the video
    pipeline submits → polls → rehosts to GCS. Only run when explicitly
    opted in via ``--include-video``."""
    print(_bold("\n[video] xAI grok-imagine-video with passthrough params"))
    if not XAI_API_KEY:
        print(_dim("  skipped: XAI_API_KEY not set"))
        return True
    print(_dim("  this can take 1-3 minutes…"))
    vid = client.videos.generate(
        prompt="a drone flying over a sunlit forest canopy",
        model="grok-imagine-video",
        duration_seconds=6,
        aspect_ratio="16:9",
        n=1,
        no_cache=True,
        no_store=True,
    )
    ok = True
    ok &= _check("got video data", bool(vid.data), f"count={len(vid.data)}")
    ok &= _check(
        "video url is an http(s) URL",
        bool(vid.data) and vid.data[0].url.startswith("http"),
        vid.data[0].url if vid.data else "<none>",
    )
    return ok


async def _run_async_smoke(api_key: str) -> bool:
    """Exercise the async SDK with one chat call to prove the ``**kwargs``
    plumbing works identically in the async code path."""
    print(_bold("\n[async] SemaCacheAsync chat.completions with passthrough params"))
    if not OPENAI_API_KEY:
        print(_dim("  skipped: OPENAI_API_KEY not set"))
        return True
    async with SemaCacheAsync(
        api_key=api_key,
        base_url=SEMCACHE_URL + "/v1",
        upstream_api_key=OPENAI_API_KEY,
    ) as client:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly: hi"}],
            temperature=0.0,
            max_completion_tokens=5,
            no_cache=True,
            no_store=True,
        )
    text = (resp.choices[0].message.content or "").strip().lower()
    return _check("async chat returned text", bool(text), f"got: {text!r}")


# ── Driver ────────────────────────────────────────────────────────────────


def _build_client(raw_key: str, upstream_key: str | None) -> SemaCache:
    kwargs: dict[str, Any] = {
        "api_key": raw_key,
        "base_url": SEMCACHE_URL + "/v1",
    }
    if upstream_key:
        kwargs["upstream_api_key"] = upstream_key
    return SemaCache(**kwargs)


def run(
    *,
    include_image: bool = False,
    include_video: bool = False,
    include_async: bool = False,
    include_params_isolation: bool = False,
) -> int:
    """Run the smoke suite and return a process-style exit code."""
    print(_bold(f"SemaCache Python SDK live smoke test"))
    print(_dim(f"  target: {SEMCACHE_URL}"))
    print(_dim(f"  user:   {TEST_USER_ID}"))

    raw_key, key_id = _mint_api_key()
    results: dict[str, bool] = {}
    try:
        client = _build_client(raw_key, OPENAI_API_KEY)
        results["chat_passthrough"] = _safe(lambda: test_chat_passthrough(client))
        results["chat_extra_body"] = _safe(lambda: test_chat_extra_body(client))
        if include_image:
            results["image_passthrough"] = _safe(lambda: test_image_passthrough(client))
        if include_params_isolation:
            results["image_params_isolation"] = _safe(
                lambda: test_image_params_isolation(client)
            )
        if include_video:
            results["video_passthrough"] = _safe(lambda: test_video_passthrough(client))
        if include_async:
            results["async_chat"] = _safe(lambda: asyncio.run(_run_async_smoke(raw_key)))
    finally:
        _delete_api_key(key_id)

    print(_bold("\nResults"))
    for name, ok in results.items():
        mark = _green("PASS") if ok else _red("FAIL")
        print(f"  [{mark}] {name}")

    failures = [name for name, ok in results.items() if not ok]
    if failures:
        print(_red(f"\n{len(failures)} test(s) failed: {', '.join(failures)}"))
        return 1
    print(_green("\nall smoke tests passed"))
    return 0


# ── pytest integration ────────────────────────────────────────────────────
#
# The functions below are picked up by pytest. They share a single API key
# per session (via a module-scoped fixture) so pytest runs don't burn through
# the per-user key limit. Image/video tests honour ``SDK_SMOKE_INCLUDE_*``
# env vars so CI can opt in selectively.


def _pytest_enabled(flag: str) -> bool:
    return os.environ.get(flag, "").lower() in ("1", "true", "yes")


try:
    import pytest
except ImportError:  # pragma: no cover — pytest is optional at runtime
    pytest = None  # type: ignore[assignment]


if pytest is not None:

    @pytest.fixture(scope="module")
    def _session_key() -> tuple[str, str | None]:
        raw_key, key_id = _mint_api_key()
        yield raw_key, key_id
        _delete_api_key(key_id)

    @pytest.fixture(scope="module")
    def sdk_client(_session_key: tuple[str, str | None]) -> SemaCache:
        raw_key, _ = _session_key
        return _build_client(raw_key, OPENAI_API_KEY)

    def test_pytest_chat_passthrough(sdk_client: SemaCache) -> None:
        assert test_chat_passthrough(sdk_client)

    def test_pytest_chat_extra_body(sdk_client: SemaCache) -> None:
        assert test_chat_extra_body(sdk_client)

    @pytest.mark.skipif(
        not _pytest_enabled("SDK_SMOKE_INCLUDE_IMAGE"),
        reason="set SDK_SMOKE_INCLUDE_IMAGE=1 to run (costs ~$0.04/run)",
    )
    def test_pytest_image_passthrough(sdk_client: SemaCache) -> None:
        assert test_image_passthrough(sdk_client)

    @pytest.mark.skipif(
        not _pytest_enabled("SDK_SMOKE_INCLUDE_PARAMS_ISOLATION"),
        reason="set SDK_SMOKE_INCLUDE_PARAMS_ISOLATION=1 to run "
               "(4 image generations, ~$0.05)",
    )
    def test_pytest_image_params_isolation(sdk_client: SemaCache) -> None:
        assert test_image_params_isolation(sdk_client)

    @pytest.mark.skipif(
        not _pytest_enabled("SDK_SMOKE_INCLUDE_VIDEO"),
        reason="set SDK_SMOKE_INCLUDE_VIDEO=1 to run (SLOW — minutes)",
    )
    def test_pytest_video_passthrough(sdk_client: SemaCache) -> None:
        assert test_video_passthrough(sdk_client)

    @pytest.mark.skipif(
        not _pytest_enabled("SDK_SMOKE_INCLUDE_ASYNC"),
        reason="set SDK_SMOKE_INCLUDE_ASYNC=1 to run the async client",
    )
    def test_pytest_async_chat(_session_key: tuple[str, str | None]) -> None:
        raw_key, _ = _session_key
        assert asyncio.run(_run_async_smoke(raw_key))


# ── CLI ───────────────────────────────────────────────────────────────────


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--include-image",
        action="store_true",
        help="also exercise images.generate (costs ~$0.04/run)",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="also exercise videos.generate (SLOW — takes minutes)",
    )
    parser.add_argument(
        "--include-async",
        action="store_true",
        help="also exercise the SemaCacheAsync client",
    )
    parser.add_argument(
        "--include-params-isolation",
        action="store_true",
        help="prove the params_hash semantic filter excludes mismatched extras "
             "(4 image generations, ~$0.05)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="shortcut for --include-image --include-video --include-async "
             "--include-params-isolation",
    )
    args = parser.parse_args()

    include_image = args.include_image or args.all
    include_video = args.include_video or args.all
    include_async = args.include_async or args.all
    include_params_isolation = args.include_params_isolation or args.all

    return run(
        include_image=include_image,
        include_video=include_video,
        include_async=include_async,
        include_params_isolation=include_params_isolation,
    )


if __name__ == "__main__":
    sys.exit(_main())
