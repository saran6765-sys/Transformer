import json
import os
from typing import Optional

def _set_env(key: str, value: Optional[str]) -> None:
    if value is None or value == "":
        return
    if not os.getenv(key):
        os.environ[key] = str(value)

def load_secrets_into_env(
    path_candidates: Optional[list] = None,
) -> None:
    """Load LLM connection details from a JSON secrets file into environment variables."""
    candidates = path_candidates or [
        os.path.join("conf", "secrets.json"),
        "secrets.json",
    ]
    for p in candidates:
        try:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            continue

        api_key = data.get("API_KEY") or data.get("api_key")
        api_base = (
            data.get("API_BASE")
            or data.get("BASE_URL")
            or data.get("api_base")
            or data.get("base_url")
        )
        model = (
            data.get("MODEL")
            or data.get("LLM_MODEL")
            or data.get("model")
            or data.get("llm_model")
        )
        provider = data.get("LLM_PROVIDER") or data.get("provider")
        deployment = data.get("DEPLOYMENT") or data.get("deployment")
        key_header = data.get("KEY_HEADER") or data.get("key_header")
        key_prefix = data.get("KEY_PREFIX") or data.get("key_prefix")
        api_version = data.get("API_VERSION") or data.get("api_version")
        system_prompt = data.get("SYSTEM_PROMPT") or data.get("system_prompt")
        extra_headers = data.get("EXTRA_HEADERS") or data.get("extra_headers")
        temperature = data.get("TEMPERATURE") or data.get("temperature")

        _set_env("LLM_API_KEY", api_key)
        _set_env("OPENAI_API_KEY", api_key)
        _set_env("LLM_BASE_URL", api_base)
        _set_env("OPENAI_BASE_URL", api_base)
        _set_env("LLM_MODEL", model)
        _set_env("OPENAI_MODEL", model)
        _set_env("LLM_PROVIDER", provider)
        _set_env("LLM_DEPLOYMENT", deployment)
        _set_env("LLM_KEY_HEADER", key_header)
        _set_env("LLM_KEY_PREFIX", key_prefix)
        _set_env("LLM_API_VERSION", api_version)
        if isinstance(extra_headers, dict):
            _set_env("LLM_EXTRA_HEADERS", json.dumps(extra_headers))
        _set_env("LLM_SYSTEM_PROMPT", system_prompt)
        _set_env("LLM_TEMPERATURE", temperature)
        break
