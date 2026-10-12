"""Chat client: OpenAI-style `/chat/completions` (DeepSeek and every
OpenAI-compatible gateway) and Anthropic-style `/messages` (Claude models)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from . import usage as _usage
from ._paths import PROJECT_ROOT


@dataclass(frozen=True)
class ChatReply:
    """One completion: its text and why the model stopped writing."""

    content: str
    finish_reason: str | None = None


def load_api_keys(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path).resolve()
    keys: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip().lstrip("\ufeff")
        if s and not s.startswith("#"):
            keys.append(s)
    if not keys:
        raise SystemExit(f"no API keys found in {p}")
    return keys


class DeepSeekClient:
    def __init__(self, api_key: str, *,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-flash",
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking_mode: str = "disabled",
                 max_attempts: int = 5,
                 backoff_seconds: List[int] | None = None,
                 extra_body: Dict[str, Any] | None = None,
                 api_style: str | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # "enabled" | "disabled" | "none" ("none" = send no thinking field;
        # gateways that do not know DeepSeek's flag reject it).
        self.thinking_mode = thinking_mode
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds or [1, 2, 4, 8, 16]
        # Provider-specific request fields merged into every payload
        # (e.g. {"enable_thinking": false} for Qwen, {"reasoning_effort": "low"}).
        self.extra_body = dict(extra_body or {})
        # "openai" = POST {base_url}/chat/completions; "anthropic" = POST
        # {base_url}/messages (Claude models on gateways that keep them there).
        self.api_style = api_style or ("anthropic" if model.lower().startswith("claude") else "openai")

    def chat(self, messages: List[Dict[str, str]],
             *, max_tokens: int | None = None,
             tag: str | None = None) -> str:
        """The reply text. Unchanged signature for every existing caller."""
        return self.chat_full(messages, max_tokens=max_tokens, tag=tag).content

    def chat_full(self, messages: List[Dict[str, str]],
                  *, max_tokens: int | None = None,
                  tag: str | None = None) -> "ChatReply":
        """Reply text *and* `finish_reason`.

        A caller that writes long documents (the distillation drafter and
        refiner) has to know whether the model stopped because it was done or
        because it hit the token ceiling — a cut-off skill document parses as
        far as it goes and ships a truncated code block. `chat` keeps
        returning a plain string so that nothing else has to change.

        `tag` labels the call in the usage log (`SONAR_USAGE_LOG`, see
        `usage.py`); without it the tag set by `usage.llm_tag` applies.
        """
        n_tokens = int(max_tokens or self.max_tokens)
        if self.api_style == "anthropic":
            url = f"{self.base_url}/messages"
            system = "\n\n".join(str(m["content"]) for m in messages if m.get("role") == "system")
            payload = {
                "model": self.model,
                "messages": [m for m in messages if m.get("role") != "system"],
                "max_tokens": n_tokens,
            }
            if system:
                payload["system"] = system
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": n_tokens,
            }
            # Every DeepSeek model needs the flag explicitly: without it the
            # server default (thinking ON) silently applies.
            if self.model.startswith("deepseek") and self.thinking_mode != "none":
                payload["thinking"] = {"type": self.thinking_mode}
        # `temperature: null` in the config omits the field (Claude 5.x rejects it).
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Cloudflare in front of some gateways rejects urllib's default UA (403, code 1010).
            "User-Agent": "sonar-ts/1.0",
        }
        if self.api_style == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                started = time.perf_counter()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read()
                    latency = time.perf_counter() - started
                    body = json.loads(raw)
                    if self.api_style == "anthropic":
                        reply, body = _from_anthropic(body)
                    else:
                        choice = body["choices"][0]
                        reply = ChatReply(
                            content=choice["message"].get("content") or "",
                            finish_reason=choice.get("finish_reason"))
                _usage.record_llm_call(       # never raises
                    body=body, model=self.model, tag=tag, latency_s=latency,
                    attempts=attempt + 1, finish_reason=reply.finish_reason)
                return reply
            except urllib.error.HTTPError as e:
                # Keep the gateway's explanation (no secrets in a response body).
                try:
                    detail = e.read().decode("utf-8", "replace")[:300].replace("\n", " ")
                except Exception:  # noqa: BLE001
                    detail = ""
                last_err = RuntimeError(f"HTTP {e.code} from {url}: {detail}")
                if e.code in (408, 429, 500, 502, 503, 504, 529):
                    delay = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                    time.sleep(delay)
                    continue
                raise last_err from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                delay = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                time.sleep(delay)
                continue
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_err}")


_ANTHROPIC_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def _from_anthropic(body: Dict[str, Any]) -> tuple["ChatReply", Dict[str, Any]]:
    """Anthropic Messages response -> (reply, OpenAI-shaped body for the usage log)."""
    if body.get("type") == "error" or "content" not in body:
        raise RuntimeError(f"Anthropic-style error body: {json.dumps(body)[:300]}")
    blocks = body.get("content") or []
    text = "".join(str(b.get("text", "")) for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text")
    stop = body.get("stop_reason")
    reply = ChatReply(content=text, finish_reason=_ANTHROPIC_STOP.get(stop, stop))
    u = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    inp = int(u.get("input_tokens") or 0)
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    prompt = inp + cache_read + cache_write
    shaped = dict(body)
    shaped["usage"] = {"prompt_tokens": prompt, "completion_tokens": out,
                       "total_tokens": prompt + out,
                       "prompt_cache_hit_tokens": cache_read}
    return reply, shaped
