import os
import json
import time
import hashlib
import re
from pathlib import Path
from urllib import request as urllib_request

from dotenv import load_dotenv

import config

load_dotenv()


class LLMClient:
    """Small adapter around Groq, Gemini, Ollama, and OpenAI-style chat APIs."""

    def __init__(self):
        self.provider = config.LLM_PROVIDER.lower()
        self.cache_dir = Path(config.CACHE_DIR) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_sec = (
            0.0 if self.provider == "ollama"
            else 60.0 / max(config.LLM_RATE_LIMIT_RPM, 1)
        )
        self.last_request_ts = 0.0

        if self.provider == "groq":
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set")
            self.client = Groq(api_key=api_key)

        elif self.provider == "gemini":
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(config.LLM_MODEL)

        elif self.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.base_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))

        elif self.provider in {"openai", "openai_compatible", "nvidia"}:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or NVIDIA_API_KEY is not set")

            base_url = os.getenv("OPENAI_BASE_URL")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = OpenAI(**client_kwargs)

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

    def generate_json_text(self, prompt):
        """Returns raw model text for a prompt."""
        cache_key = self._cache_key(prompt)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        raw = self._with_retries(prompt)
        normalized = self._normalize_json_text(raw)
        self._write_cache(cache_key, normalized)
        return normalized

    def _with_retries(self, prompt):
        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                self._throttle()
                return self._generate_uncached(prompt)
            except Exception as exc:
                last_error = exc
                if not self._is_rate_limit_error(exc):
                    raise

                sleep_sec = config.LLM_RETRY_BASE_SEC * (2 ** attempt)
                print(f"  Warning: LLM rate-limited, retrying in {sleep_sec:.1f}s")
                time.sleep(sleep_sec)

        raise last_error

    def _generate_uncached(self, prompt):
        """Performs a live request without cache handling."""

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        if self.provider == "ollama":
            payload = {
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                }
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(
                f"{self.base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib_request.urlopen(
                req, timeout=config.OLLAMA_TIMEOUT_SEC
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
            return str(data.get("response", "")).strip()

        if self.provider in {"openai", "openai_compatible", "nvidia"}:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        generation_config = {
            "temperature": config.LLM_TEMPERATURE,
            "max_output_tokens": config.LLM_MAX_TOKENS,
            "response_mime_type": "application/json",
        }
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return (response.text or "").strip()

    def _throttle(self):
        now = time.time()
        wait_for = self.min_interval_sec - (now - self.last_request_ts)
        if wait_for > 0:
            time.sleep(wait_for)
        self.last_request_ts = time.time()

    def _cache_key(self, prompt):
        payload = {
            "provider": self.provider,
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "prompt": prompt,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest

    def _read_cache(self, cache_key):
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("response")

    def _write_cache(self, cache_key, response_text):
        path = self.cache_dir / f"{cache_key}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"response": response_text}, handle, ensure_ascii=False)

    def _normalize_json_text(self, response_text):
        """
        Normalizes model output before JSON parsing.

        Local models sometimes wrap valid JSON in Markdown fences like:
        ```json
        {...}
        ```
        The decomposer and reasoner expect raw JSON text, so strip the
        fence wrapper while preserving the inner payload.
        """

        text = (response_text or "").strip()

        fenced = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if fenced:
            return fenced.group(1).strip()

        if text.lower().startswith("json"):
            remainder = text[4:].lstrip(" \t\r\n:")
            if remainder.startswith("{") or remainder.startswith("["):
                return remainder

        return text

    def _is_rate_limit_error(self, exc):
        message = str(exc).lower()
        return (
            "429" in message or
            "rate limit" in message or
            "too many requests" in message or
            "quota exceeded" in message
        )
