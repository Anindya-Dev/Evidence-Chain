import os

from dotenv import load_dotenv

import config

load_dotenv()


class LLMClient:
    """Small adapter around Groq and Gemini chat-style generation."""

    def __init__(self):
        self.provider = config.LLM_PROVIDER.lower()

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

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

    def generate_json_text(self, prompt):
        """Returns raw model text for a prompt."""

        if self.provider == "groq":
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
