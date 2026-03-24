import os
import json
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv

import config

load_dotenv()


class LLMClient:
    """Small adapter around Groq, Gemini, and OpenAI-compatible chat APIs."""

    def __init__(self):
        self.provider = config.LLM_PROVIDER.lower()
        self.cache_dir = Path(config.CACHE_DIR) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_sec = 60.0 / max(config.LLM_RATE_LIMIT_RPM, 1)
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
        self._write_cache(cache_key, raw)
        return raw

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

    def _is_rate_limit_error(self, exc):
        message = str(exc).lower()
        return (
            "429" in message or
            "rate limit" in message or
            "too many requests" in message or
            "quota exceeded" in message
        )
