"""Simple HTTP chat LLM client for OpenAI-compatible REST endpoints."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import httpx


class DirectChatLLM:
    """Minimal chat client that exposes a LangChain-like ``predict`` method.

    Designed for custom gateways where the API key is provided via a header such
    as ``X-Api-Key`` and the endpoint follows the OpenAI chat completions schema.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        deployment: Optional[str] = None,
        key_header: Optional[str] = None,
        key_prefix: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for DirectChatLLM")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.deployment = deployment
        self.key_header = key_header or ("Authorization" if (key_prefix or not deployment) else "api-key")
        self.key_prefix = key_prefix
        self.api_version = api_version
        self.temperature = temperature
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.system_prompt = system_prompt

    def _build_url(self) -> str:
        if self.deployment:
            path = f"/deployments/{self.deployment}/chat/completions"
        else:
            path = "/chat/completions"
        url = f"{self.base_url}{path}"
        if self.api_version:
            joiner = "?" if "?" not in url else "&"
            url = f"{url}{joiner}api-version={self.api_version}"
        return url

    def predict(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.model:
            payload["model"] = self.model

        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        if self.api_key:
            token = self.api_key
            if self.key_prefix:
                token = f"{self.key_prefix.strip()} {self.api_key}".strip()
            headers[self.key_header] = token

        url = self._build_url()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as err:
            body = err.response.text
            raise RuntimeError(f"LLM call failed: {err.response.status_code} {body}") from err
        except httpx.HTTPError as err:
            raise RuntimeError(f"LLM network error: {err}") from err

        try:
            result = response.json()
        except json.JSONDecodeError as err:
            raise RuntimeError("Invalid JSON response from LLM endpoint") from err

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise RuntimeError(f"Unexpected LLM response schema: {result}") from err

        return content or ""


def from_env() -> DirectChatLLM:
    """Factory helper that constructs a ``DirectChatLLM`` from environment variables."""

    extra_headers: Dict[str, str] = {}
    headers_json = os.getenv("LLM_EXTRA_HEADERS")
    if headers_json:
        try:
            maybe_dict = json.loads(headers_json)
            if isinstance(maybe_dict, dict):
                extra_headers = {str(k): str(v) for k, v in maybe_dict.items()}
        except json.JSONDecodeError:
            pass

    return DirectChatLLM(
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "",
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL"),
        deployment=os.getenv("LLM_DEPLOYMENT"),
        key_header=os.getenv("LLM_KEY_HEADER"),
        key_prefix=os.getenv("LLM_KEY_PREFIX"),
        api_version=os.getenv("LLM_API_VERSION"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        timeout=float(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "60"))),
        extra_headers=extra_headers,
        system_prompt=os.getenv("LLM_SYSTEM_PROMPT"),
    )

