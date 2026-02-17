from __future__ import annotations

from dataclasses import dataclass
import time

import requests

from .settings import settings


@dataclass
class OllamaModelChoice:
    name: str
    reason: str


@dataclass
class InferenceResult:
    response: dict
    model_used: str
    profile_used: str
    latency_seconds: float
    fallback_used: bool


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: int | None = None) -> None:
        self.base_url = base_url or settings.ollama_base_url
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json() or {}
        models = payload.get("models", [])
        return [model.get("name", "") for model in models if model.get("name")]

    def select_best_model(self, installed: list[str]) -> OllamaModelChoice:
        if settings.ollama_model:
            return OllamaModelChoice(settings.ollama_model, "Explicit model set in environment")

        if settings.ollama_low_memory_mode:
            priorities = [
                "qwen2.5:14b",
                "qwen2.5",
                "deepseek-r1",
                "llama3.1:8b",
                "llama3.2",
                "llama3.3:70b",
            ]
        else:
            priorities = [
                "deepseek-r1",
                "qwen2.5:32b",
                "qwen2.5:14b",
                "qwen2.5",
                "llama3.3:70b",
                "llama3.1:8b",
                "llama3.2",
            ]

        for target in priorities:
            exact = [name for name in installed if name == target]
            if exact:
                return OllamaModelChoice(exact[0], f"Matched priority model {target}")

            partial = [name for name in installed if name.startswith(target + ":") or target in name]
            if partial:
                return OllamaModelChoice(partial[0], f"Matched compatible priority model for {target}")

        fallback = installed[0] if installed else ""
        if fallback:
            return OllamaModelChoice(fallback, "Using first installed model as fallback")

        raise RuntimeError("No Ollama models available. Pull at least one model with `ollama pull <model>`.")

    def rank_models(self, installed: list[str]) -> list[str]:
        primary = self.select_best_model(installed).name
        ordered = [primary]
        for model in installed:
            if model not in ordered:
                ordered.append(model)
        return ordered

    def _profile_options(self, profile: str) -> dict:
        base_gpu_layers = settings.ollama_gpu_layers if settings.prefer_gpu else 0
        base_ctx = settings.ollama_num_ctx
        base_predict = settings.ollama_num_predict

        if profile == "base":
            return {
                "num_gpu": base_gpu_layers,
                "num_ctx": base_ctx,
                "num_predict": base_predict,
                "keep_alive": settings.ollama_keep_alive,
            }

        if profile == "reduced":
            return {
                "num_gpu": max(0, int(base_gpu_layers * 0.75)),
                "num_ctx": max(512, base_ctx // 2),
                "num_predict": max(128, base_predict // 2),
                "keep_alive": "2m",
            }

        return {
            "num_gpu": max(0, int(base_gpu_layers * 0.5)),
            "num_ctx": max(512, base_ctx // 4),
            "num_predict": max(96, base_predict // 4),
            "keep_alive": "1m",
        }

    def _generate_with_profile(self, model: str, prompt: str, profile: str) -> tuple[dict, float]:
        options = self._profile_options(profile)
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": options["keep_alive"],
            "options": {
                "num_gpu": options["num_gpu"],
                "num_ctx": options["num_ctx"],
                "num_predict": options["num_predict"],
                "temperature": 0.2,
            },
        }

        start = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        latency = time.perf_counter() - start
        return response.json(), latency

    def generate_with_fallback(self, ranked_models: list[str], prompt: str) -> InferenceResult:
        if not ranked_models:
            raise RuntimeError("No ranked models provided for inference")

        profiles = ["base", "reduced", "minimal"]
        max_attempts = min(settings.ollama_max_retries_per_symbol, len(ranked_models) * len(profiles))
        attempts = 0
        last_error: Exception | None = None

        for model in ranked_models:
            for profile in profiles:
                attempts += 1
                if attempts > max_attempts:
                    break

                try:
                    raw, latency = self._generate_with_profile(model, prompt, profile)
                    too_slow = latency > settings.ollama_latency_threshold_seconds

                    if too_slow and profile != "minimal":
                        continue

                    return InferenceResult(
                        response=raw,
                        model_used=model,
                        profile_used=profile,
                        latency_seconds=latency,
                        fallback_used=attempts > 1 or profile != "base",
                    )
                except Exception as exc:
                    last_error = exc

            if attempts > max_attempts:
                break

        if last_error:
            raise RuntimeError(f"All fallback attempts failed: {last_error}") from last_error
        raise RuntimeError("All fallback attempts failed without a specific exception")

    def generate_json(self, model: str, prompt: str) -> dict:
        raw, _ = self._generate_with_profile(model, prompt, "base")
        return raw

    def generate_fast(self, model: str, prompt: str, max_tokens: int = 256) -> InferenceResult:
        """Single-shot generate with no latency checking or fallback.

        Ideal for council agents where we need fast JSON responses and
        don't want wasted retries from the latency threshold.
        """
        options = {
            "num_gpu": settings.ollama_gpu_layers if settings.prefer_gpu else 0,
            "num_ctx": min(settings.ollama_num_ctx, 2048),
            "num_predict": max_tokens,
            "keep_alive": settings.ollama_keep_alive,
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": options["keep_alive"],
            "options": {
                "num_gpu": options["num_gpu"],
                "num_ctx": options["num_ctx"],
                "num_predict": options["num_predict"],
                "temperature": 0.3,
            },
        }

        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        latency = time.perf_counter() - start

        return InferenceResult(
            response=response.json(),
            model_used=model,
            profile_used="fast",
            latency_seconds=latency,
            fallback_used=False,
        )
