# Audit & Improvements Report

**Codebase:** `auto-gui` — pluggable visual-grounding GUI automation agent
**Scope:** Stage 7 audit, bug fixes, test coverage expansion, feature extension

---

## 1. Bugs found and how they were resolved

### 1.1 Critical: corrupted image payloads sent to the grounding model

**File:** `src/grounding/providers/vlmrun.py`

Two independent bugs were stacked on top of each other in `_build_messages`,
and either one alone would have broken every grounding request:

1. `vlmrun.common.image.encode_image()` (from the VLM Run SDK) already
   returns a **complete** `data:image/png;base64,<payload>` URI, not a bare
   base64 string. The provider code re-wrapped that return value in a second,
   hardcoded `data:image/png;base64,` prefix, producing a doubly-prefixed,
   undecodable URL (e.g. `data:image/png;base64,data:image/png;base64,...`).
2. The (already-corrupted) payload was then embedded using Python's `!r`
   conversion (`f"...{base64_str!r}"`), which wraps the value in `'...'` or
   `b'...'` repr quoting/escaping — corrupting it a second time over.

**Impact:** every single call to the `vlmrun` grounding provider sent an
image the model could not decode. This would have surfaced as either silent
grounding failures or nonsensical detections in production, and would not
have been caught by any existing test, since no test exercised
`_build_messages` end-to-end.

**Fix:** `_encode_image` now documents and returns the SDK's data URL as-is;
`_build_messages` uses it directly instead of re-prefixing it, and only
decodes `bytes` → `str` when necessary (base64 output is always ASCII-safe).

**Regression tests:** `tests/test_vlmrun_bugfix.py` — asserts the URL has
exactly one `data:image/` prefix and one `;base64,` marker, contains no
repr-style quoting, and that the payload round-trips through
`base64.b64decode` back into a valid image of the original dimensions.

---

### 1.2 High-DPI / Retina coordinate scaling bug (explicitly called out in the task brief)

**Files:** `src/agent/tools/action/actions.py`, `src/agent/tools/coordinates.py` (new)

Screenshots — and therefore every bounding box a grounding provider
returns — are captured in **physical pixel** space. `pyautogui`'s
mouse/keyboard input APIs (`moveTo`, `click`, `dragTo`, ...), however,
operate in the **logical** OS coordinate space reported by
`pyautogui.size()`. On any display with a scale factor other than 1.0 (2x
Retina, 4K with 150%/200% Windows scaling, mixed-DPI multi-monitor setups),
these two spaces diverge, and a bounding-box-derived click coordinate passed
straight to `pyautogui.click(x, y)` lands in the wrong place — or off-screen
entirely on a small enough logical canvas.

`ActionManager._resolve_position` previously did exactly that: it computed a
bbox/detection center in screenshot pixel space and dispatched it directly to
`pyautogui`, with no scaling step anywhere in the codebase.

**Fix:** added `CoordinateMapper` (`src/agent/tools/coordinates.py`), which
computes the physical→logical scale factor from a screenshot backend and
`pyautogui.size()`, caches it, and converts screenshot-space coordinates to
logical coordinates. Wired into `ActionManager` (constructor accepts an
optional `coordinate_mapper`; defaults to a real one backed by
`pyautogui.screenshot()`/`pyautogui.size()`). On unscaled displays the scale
factor is exactly `1.0` and the mapper is a no-op, so behavior on the common
case is unchanged.

**Tests:** `tests/test_coordinates.py` (scale computation, caching, 1x/2x/
asymmetric-scale conversion, rounding) and `tests/test_actions.py`
(`ActionManager` dispatches DPI-corrected coordinates to `pyautogui` for
tuple, `BoundingBox`, and `GroundingDetection` targets).

---

### 1.3 Silent error handling / no structured logging

**Files:** `src/agent/tools/windows.py`, `src/agent/agent.py`, plus new
`src/grounding/logging_utils.py` / `src/agent/logging_config.py`

- `windows.py` swallowed window-restore failures with a bare `print()`,
  invisible in any log aggregation setup and impossible to filter by level.
- `agent.py`'s `run()`/`arun()` had no top-level exception handling around
  the LangGraph invocation. Any failure (model backend error, unexpected
  tool exception escaping `ToolNode`, context construction failure) crashed
  with a raw traceback and no structured record of what task was running or
  how far it got.

**Fix:** added a centralized, environment-configurable `logging` setup
(`GUI_AGENT_LOG_LEVEL`, `GUI_AGENT_LOG_FILE`, `GUI_AGENT_LOG_FORMAT`) shared
by both the `grounding` and `agent` packages under a single `auto_gui.*`
logger hierarchy. Replaced the `print()` in `windows.py` with a structured
`logger.warning(...)` call. Wrapped `run()`/`arun()` in try/except that logs
the failure with full context (goal, exception type, stack trace) and
re-raises as a new `AgentExecutionError` (`src/agent/exceptions.py`) instead
of an opaque crash.

**Tests:** `tests/test_agent.py` (`AgentExecutionError` is raised with the
original exception as `__cause__` and the task goal attached, for both
`run()` and `arun()`); `tests/test_windows.py` (restore failures don't
propagate and are logged).

**This paid off directly during real-world testing** — see §1.6 and §1.7
below, both of which were discovered from clean, actionable
`AgentExecutionError` output on a live CLI run instead of a raw crash.

---

### 1.4 Non-hermetic tests that depended on live network access / real credentials

**File:** `tests/integration/test_vlmrun_provider.py`

`test_initialize_creates_client` constructed a real `VLMRun` SDK client with
no mocking. The SDK's constructor performs a live health-check network call,
so this test either required a real `VLMRUN_API_KEY` in the ambient
environment to pass, or failed with a `403 Forbidden` against the real VLM
Run API — discovered while getting the suite to run fully offline. This is
exactly the class of problem the task's "mock fixtures... to allow full
offline test execution" requirement calls out, just found in the test suite
itself rather than application code.

**Fix:** patched the `VLMRun` SDK class in the test so `_initialize()` is
exercised with zero network I/O, and gave the `provider` fixture a dummy API
key so it never depends on ambient credentials.

---

### 1.5 Minor issues

- **`GroundingFallbackError` defined but unused** (`src/grounding/exceptions.py`):
  the exception type for a fallback strategy existed with no code path that
  ever raised it — a strong signal the fallback feature (task requirement
  4) was designed for but never implemented. See §3.
- **`LocateToolInput.provider` was `Field(...)` (required) despite being
  typed `str | None`** (`src/agent/tools/tool_models.py`): forced every tool
  call to explicitly pass `provider` (even as `null`) instead of omitting it.
  Changed to `Field(default=None, ...)`.
- **Malformed VLM Run detection payloads were silently dropped**
  (`src/grounding/providers/vlmrun.py`): items in the model's response that
  failed bbox/type validation were `continue`d past with no record. Now
  counted and logged via `logger.warning` with counts of skipped vs.
  accepted detections, so a model that starts returning malformed payloads
  is visible in logs instead of silently degrading.

---

### 1.6 `capture_screen` tool had no explicit `args_schema` (found via live CLI run)

**Files:** `src/agent/tools/tool_models.py`, `src/agent/tools/agent_tools.py`

Surfaced after this audit's initial pass, running the CLI end-to-end against
a real chat model:

```
pydantic.errors.PydanticInvalidForJsonSchema: Failed to generate JSON
schema for 'capture_screen': Cannot generate a JsonSchema for
core_schema.IsInstanceSchema (<class 'PIL.Image.Image'>)
```

`capture_screen` was the only tool in `GUI_AGENT_TOOLS` registered without
an explicit `args_schema`. Its function's only parameter is the injected
`Runtime[AgentContext]`, so LangChain fell back to auto-inferring a schema
from the function signature — which walked into `AgentContext` and hit
`ScreenshotResult.image: PIL.Image.Image`, an arbitrary type Pydantic
cannot render as JSON Schema. This broke `model.bind_tools(GUI_AGENT_TOOLS)`
for **every** tool the moment `capture_screen` was included in the list —
not just `capture_screen` itself.

Notably, the main-loop error handling added in §1.3 worked exactly as
intended here: instead of a raw, unhandled traceback, the CLI printed a
clean `Error: Agent run failed while executing task '...'` with the root
cause, from a logged `AgentExecutionError`.

**Why this wasn't caught by the earlier test pass:** `tests/test_agent.py`'s
fake chat models override `bind_tools()` entirely (to avoid needing a real
model backend for basic run/exception-handling tests), which bypasses the
real JSON-schema-conversion code path this bug lives in.

**Fix:** added an explicit, empty `CaptureScreenToolInput` schema, matching
every other tool in the module.

**Regression tests:** `tests/test_tool_schemas.py` — exercises the real
`langchain_core.utils.function_calling.convert_to_openai_tool` conversion
(the same one `bind_tools()` uses) against every tool in `GUI_AGENT_TOOLS`,
with no chat model involved, so this class of bug can't hide behind a test
double again. Also asserts no tool ships without an explicit `args_schema`,
guarding against a future tool reintroducing the same failure mode.

---

### 1.7 Unbounded `max_tokens` caused OpenRouter 402s (found via live CLI run)

**Files:** `src/agent/config.py`, `src/agent/vlm.py`

Surfaced immediately after fixing §1.6, on the next live CLI run:

```
openai.APIStatusError: Error code: 402 - This request requires more
credits, or fewer max_tokens. You requested up to 65536 tokens, but
can only afford 2097. To increase, visit
https://openrouter.ai/settings/credits...
```

`create_vlm()` constructed `ChatOpenAI` with no `max_tokens` at all.
OpenAI-compatible gateways such as OpenRouter interpret an unset
`max_tokens` as "reserve up to the model's full context window" for
every single request — regardless of how many output tokens a turn
actually needs. A GUI-automation tool-calling turn needs nowhere near
65536 output tokens, so this failed outright on any account without a
large credit balance, independent of whether the account is otherwise
healthy.

**Fix:** added a configurable `AgentSettings.max_tokens`
(`AGENT_MAX_TOKENS`, default `4096`), wired through `create_vlm()`.
Settable to `null`/unset to restore the old unbounded behavior if a
given backend needs it. Documented in `.env.example`.

**Tests:** `tests/test_vlm.py` — asserts `create_vlm()` passes the
configured value through, that it can be explicitly disabled, and
that the default is a positive, bounded value.

---

## 2. Test coverage: before vs. after

Coverage measured with `pytest --cov=src --cov-report=term-missing`.

| | Before | After |
|---|---|---|
| Test files | 2 test modules (`tests/test_registry.py`, `tests/test_client.py`, `tests/test_client_async.py`, `tests/integration/*`) — **grounding package only** | 16 test modules covering both `grounding` and `agent` packages |
| Test count | 94 | **364** |
| `src/agent/*` coverage | **0%** (untested; every module imports `pyautogui`/`pywinctl`, which require a live display and previously made the package uncollectable in headless CI) | **~85-95%** across `agent.py`, `actions.py`, `coordinates.py`, `screenshot.py`, `windows.py`; `agent_tools.py` ~71% |
| `src/grounding/*` coverage | ~70-85% on individual modules, **32% overall** (client fallback path didn't exist yet; several branches in `base.py`/`interfaces.py`/`vlmrun.py` untested) | **~78-98%** across modules |
| **Overall (`src/`)** | **32%** | **82%** |

**Known remaining gaps** (called out explicitly rather than hidden):

- `src/agent/cli.py` (0%) — the CLI entrypoint wasn't covered; it's thin
  argument-parsing glue around `GUIAutomationAgent`, lower risk than the
  library code, but should get smoke tests (e.g. via `click`'s
  `CliRunner` or `argparse` invocation with a stubbed agent) before a `1.0`.
- `src/agent/tools/action/actions_async.py` (0%) — an async mirror of
  `actions.py`. It isn't wired into any current tool or code path (`grep`
  shows no callers), so it's untested and, more importantly, **unverified
  that it stays behaviorally in sync with `actions.py`** (e.g. it predates
  the DPI coordinate-mapping fix in this audit and does not yet use
  `CoordinateMapper`). Recommend either deleting it if truly unused, or
  giving it the same test treatment as `actions.py` and wiring it into the
  async tool path if it's meant to be used.

### Why the `agent` package was untestable before this audit

`pyautogui` and `pywinctl` require a live display/session just to *import*
(`mouseinfo`/`pymsgbox` probe `$DISPLAY` at import time on Linux). This is
almost certainly why the pre-existing test suite covered only `grounding`
(pure Python, no GUI dependency) and left the entire `agent` package at 0%.

Rather than depending on a virtual framebuffer (Xvfb) — which adds CI
complexity, is Linux-only, and still doesn't help on other platforms — this
audit added lightweight, deterministic fakes for `pyautogui`/`pywinctl`
(`tests/agent_fakes.py`) that are installed into `sys.modules` before test
collection (`conftest.py`), following the same "mock the backend to allow
offline execution" principle the codebase already applies to grounding
providers. This makes the entire suite platform-independent and fast
(**364 tests in under 15 seconds**, no display, no network, no GPU).

---

## 3. Feature extension: fallback/retry grounding strategy

Requirement 4 asked for a fallback strategy ("if a high-speed smaller
grounding model yields confidence below a threshold, fall back to a
higher-capacity vision backend") and runtime metrics for latency/confidence.
The unused `GroundingFallbackError` (see §1.5) made clear where this was
meant to live.

**`GroundingClient.locate_with_fallback()` / `.alocate_with_fallback()`**
(`src/grounding/client.py`) — tries a sequence of provider ids in order,
advancing to the next provider when the current one:

- raises a `GroundingProviderError`,
- returns no usable detections, or
- returns a best-detection confidence below an optional `min_confidence`
  threshold.

Raises `GroundingFallbackError` (carrying every per-provider failure reason)
only if *all* configured providers are exhausted. Example:

```python
response = client.locate_with_fallback(
    providers=["fast-local-model", "vlmrun"],
    image=screenshot,
    query="Submit button",
    min_confidence=0.6,
)
```

**`click_target` agent tool** (`src/agent/tools/agent_tools.py`) — the
"retry logic with dynamic screen re-capture on click failure" half of
requirement 4. Combines locate + click + visual verification:

1. Capture the screen and locate the described element.
2. Click its center (through the DPI-corrected coordinate path, §1.2).
3. **Re-capture the screen and re-locate the same query.** If an element
   at effectively the same position is still detected, that's treated as
   strong evidence the click didn't register (dialog didn't open, button
   didn't respond, etc.), and the tool retries — from a fresh screenshot —
   up to `max_attempts` times.
4. If the element is gone or has moved, the click is treated as
   successful.

This is registered in `GUI_AGENT_TOOLS` alongside the existing `click`
tool, so the agent can choose the self-correcting variant for
higher-stakes clicks.

**Metrics:** `ActionResult` gained a `latency_ms` field, populated by
`ActionManager._make_result` and emitted via structured `logger.info` on
every `click`/`move_mouse`/`drag`, giving per-action execution latency.
Grounding provider inference latency was already tracked in
`GroundingResponse` metadata; the `vlmrun` provider's `_submit_prediction`
now also logs it directly via `perf_counter()`-based timing.

Tests: `tests/test_client_fallback.py` (9 tests — first-provider success,
fallback-on-exception, fallback-on-empty-result, fallback-on-low-confidence,
all-providers-exhausted, empty-provider-list validation, sync + async);
`tests/test_agent_tools.py` (`click_target` retry/re-capture behavior,
including that a screenshot is taken on every attempt).

---

## 4. Proposed structural improvements

### 4.1 Additional grounding backend providers

- **Provider interface is already clean** (`BaseGroundingProvider` /
  `GroundingEngine`) — adding a new backend (e.g. a local open-weight
  model, or a second hosted API) is a matter of implementing `_locate`/
  `_alocate` and registering the class. No changes needed here; the
  registry + fallback client built in §3 already support an arbitrary
  provider chain.
- **Recommend a `LocalGroundingProvider` base** built on
  `LocalProviderSettings` (already defined in `config.py` but currently
  unused by any concrete provider) so a small local model can serve as the
  fast "first hop" in a fallback chain, with `vlmrun` as the accurate
  fallback — matching the exact pattern requirement 4 describes.
- **Provider health/warm-up hook:** `requires_initialization` exists but
  there's no way to eagerly warm up (e.g. load model weights, open a
  connection pool) before the first real request on the critical path.
  A `GroundingClient.warm_up(provider_ids)` that calls `_initialize()`
  without a real request would remove first-call latency spikes.

### 4.2 Inference latency reduction

- **Screenshot resolution vs. model input size:** full-desktop screenshots
  (e.g. 3840x2160 on 4K) are sent to the model as-is. Most grounding models
  have a fixed or capped input resolution; downscaling before encoding
  (with bbox coordinates scaled back up afterward) would cut both upload
  size and inference time without losing accuracy, since UI text/icons stay
  legible well below native 4K. This composes naturally with the new
  `CoordinateMapper` — it already centralizes coordinate-space conversion,
  so a "screenshot space -> model input space" mapping is a small extension
  of the same abstraction rather than a new one.
- **Provider warm-up (see 4.1)** avoids paying cold-start latency on the
  first tool call of every run.
- **Refresh the coordinate mapper on display changes:** the scale factor is
  cached until `refresh()` is called explicitly; nothing currently triggers
  that after e.g. `maximize_window`/a monitor change mid-run. Worth wiring
  up if multi-monitor support becomes a priority.
- **Parallel provider racing** as an alternative to sequential fallback:
  for latency-sensitive flows, `asyncio.wait(..., return_when=FIRST_COMPLETED)`
  across 2 providers with the same query, taking whichever returns first
  with acceptable confidence, would trade cost for latency where that's
  the priority. `alocate_with_fallback`'s structure makes this a relatively
  small follow-up change.
- **Bound output tokens per turn (see §1.7).** Beyond just avoiding 402s,
  a tighter, empirically-tuned `AGENT_MAX_TOKENS` also caps worst-case
  per-turn latency, since token generation time scales with the cap even
  when the model stops early most of the time.

### 4.3 Other observations for future work

- Consider wiring `CoordinateMapper.refresh()` into the `capture_screen`
  tool so display-configuration changes mid-run are picked up automatically
  rather than relying on the initial cached scale factor for the whole run.
- `actions_async.py` should either be deleted (if genuinely unused) or
  brought back in sync with `actions.py` (DPI mapping, latency tracking)
  and covered by tests — right now it's a maintenance trap: a second
  implementation of the same action surface that can silently drift.
- `src/agent/cli.py` would benefit from the same offline-testable treatment
  applied here to `agent.py`/`actions.py` — smoke tests with a stubbed
  `GUIAutomationAgent` are cheap and would catch entrypoint regressions.
- Consider a `tests/test_tool_schemas.py`-style "does this actually bind to
  a real chat model's schema converter" smoke test as a standing CI gate,
  not just a one-off regression test — it's cheap (no network/model calls)
  and would have caught §1.6 before it ever reached a live run.

---

## 5. Summary

| Deliverable | Status |
|---|---|
| Bug audit & fixes | 7 bugs fixed (2 critical: image corruption, DPI coordinates; 2 found via live CLI runs: tool schema, unbounded max_tokens; 1 test-hermeticity; 2 minor), all with regression tests |
| Test coverage expansion | 94 -> 364 tests; 32% -> 82% overall `src/` coverage; `agent` package taken from untestable (0%) to ~85%+ on core modules |
| Feature extension | Fallback/retry grounding strategy (`locate_with_fallback`) + resilient `click_target` tool with dynamic re-capture retry |
| Structured logging & observability | Centralized, env-configurable `logging` setup shared across both packages; latency/confidence logging on grounding + action execution |
| `.env.example` | Added, documenting every `AGENT_*`, `GROUNDING_*`, `VLMRUN_*`, and `GUI_AGENT_LOG_*` variable |