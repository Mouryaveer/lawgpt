"""Direct Groq client for the configured Qwen3-32B generation model."""

from __future__ import annotations

import json
import requests

from .prompts import GROUNDING_SYSTEM_PROMPT, make_user_prompt


class GroqGenerationError(RuntimeError):
    pass


class GroqGenerator:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 45.0):
        if not api_key:
            raise GroqGenerationError("GROQ_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, query: str, evidence: list[dict]) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0.1, "messages": [{"role": "system", "content": GROUNDING_SYSTEM_PROMPT}, {"role": "user", "content": make_user_prompt(query, evidence)}], "response_format": {"type": "json_object"}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str) or not isinstance(parsed.get("claims", []), list):
                raise GroqGenerationError("Generation provider returned an invalid structured answer")
            return parsed
        except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise GroqGenerationError("Generation provider request failed") from exc
