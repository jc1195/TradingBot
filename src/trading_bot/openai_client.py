from __future__ import annotations

from dataclasses import dataclass
import json
import time

import requests

from .ollama_client import InferenceResult
from .settings import settings


@dataclass
class OpenAIModelChoice:
    name: str
    reason: str


class OpenAIClient:
    def __init__(self) -> None:
        self.base_url = settings.openai_base_url.rstrip("/")
        self.timeout_seconds = settings.openai_timeout_seconds

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key.strip())

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        if not self.is_configured():
            return []

        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json() or {}
            items = payload.get("data", [])
            return [item.get("id", "") for item in items if item.get("id")]
        except Exception:
            return [settings.openai_model]

    def select_best_model(self, installed: list[str]) -> OpenAIModelChoice:
        if settings.openai_model:
            return OpenAIModelChoice(settings.openai_model, "Configured OpenAI model")

        priorities = ["gpt-5", "gpt-5.1", "gpt-4.1", "gpt-4o"]
        for target in priorities:
            for model in installed:
                if model == target or model.startswith(target):
                    return OpenAIModelChoice(model, f"Matched priority OpenAI model {target}")

        fallback = installed[0] if installed else "gpt-5"
        return OpenAIModelChoice(fallback, "Using fallback OpenAI model")

    def rank_models(self, installed: list[str]) -> list[str]:
        primary = self.select_best_model(installed).name
        ranked = [primary]
        for model in installed:
            if model and model not in ranked:
                ranked.append(model)
        return ranked

    def _profile_max_tokens(self, profile: str) -> int:
        if profile == "base":
            return settings.openai_max_output_tokens
        if profile == "reduced":
            return max(256, settings.openai_max_output_tokens // 2)
        return max(128, settings.openai_max_output_tokens // 4)

    def _extract_text_response(self, payload: dict) -> str:
        choices = payload.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                return "\n".join(chunks)
        return "{}"

    def _generate_with_profile(self, model: str, prompt: str, profile: str) -> tuple[dict, float]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only strict JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": self._profile_max_tokens(profile),
        }

        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        raw = response.json() or {}
        text_response = self._extract_text_response(raw)
        return {"response": text_response, "raw": raw}, elapsed

    def generate_with_fallback(self, ranked_models: list[str], prompt: str) -> InferenceResult:
        if not self.is_configured():
            raise RuntimeError("OpenAI API key is not configured")
        if not ranked_models:
            raise RuntimeError("No OpenAI models available for inference")

        profiles = ["base", "reduced", "minimal"]
        max_attempts = min(settings.openai_max_retries_per_symbol, len(ranked_models) * len(profiles))
        attempts = 0
        last_error: Exception | None = None

        for model in ranked_models:
            for profile in profiles:
                attempts += 1
                if attempts > max_attempts:
                    break
                try:
                    response, latency = self._generate_with_profile(model, prompt, profile)
                    too_slow = latency > settings.ollama_latency_threshold_seconds
                    if too_slow and profile != "minimal":
                        continue

                    return InferenceResult(
                        response=response,
                        model_used=model,
                        profile_used=f"openai_{profile}",
                        latency_seconds=latency,
                        fallback_used=attempts > 1 or profile != "base",
                    )
                except Exception as exc:
                    last_error = exc

            if attempts > max_attempts:
                break

        if last_error:
            raise RuntimeError(f"OpenAI fallback attempts failed: {last_error}") from last_error
        raise RuntimeError("OpenAI fallback attempts failed without a specific exception")
