"""DeepSeek chat client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from ._paths import PROJECT_ROOT


def load_api_keys(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path).resolve()
    keys: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s.startswith("sk-"):
            keys.append(s)
    if not keys:
        raise SystemExit(f"no API keys found in {p}")
    return keys


class DeepSeekClient:
    def __init__(self, api_key: str, *,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-v4-flash",
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking_mode: str = "disabled",
                 max_attempts: int = 5,
                 backoff_seconds: List[int] | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode  # "enabled" | "disabled"
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds or [1, 2, 4, 8, 16]

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model.startswith("deepseek-v4"):
            payload["thinking"] = {"type": self.thinking_mode}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read())
                    return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    delay = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                    time.sleep(delay)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                delay = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                time.sleep(delay)
                continue
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_err}")
