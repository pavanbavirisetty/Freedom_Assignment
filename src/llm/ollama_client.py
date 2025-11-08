from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from src.utils.config import NewsArticle, ViralIdea, get_settings
from src.utils.logger import get_logger


class OllamaClient:
    """Simple wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.logger = get_logger(self.__class__.__name__)
        self.client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(60.0))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, **options: Any) -> str:
        """Call Ollama's generate endpoint and return concatenated text."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        for key, value in options.items():
            if key == "temperature":
                payload.setdefault("options", {})["temperature"] = value
            else:
                payload[key] = value

        self.logger.debug("Sending prompt to Ollama (model=%s)", self.model)
        response = self.client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def chat(self, messages: List[Dict[str, str]], **options: Any) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
        }
        for key, value in options.items():
            if key == "temperature":
                payload.setdefault("options", {})["temperature"] = value
            else:
                payload[key] = value
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def close(self) -> None:
        self.client.close()


def build_viral_idea_prompt(article: NewsArticle) -> str:
    template = f"""
You are a social media strategist. Produce a single viral Instagram carousel concept based on the news article.

Return ONLY a compact JSON object with the keys:
- "headline": short captivating title (string)
- "angle": unique hook for the post (string)
- "key_points": array of 3-5 bullet strings describing slides or talking points

Do not add markdown, backticks, explanations, or multiple JSON objects. If data is missing, infer creatively.

Article title: {article.title}
Summary: {article.summary or "N/A"}
Source URL: {article.url}
Published at: {article.published_at or "N/A"}
"""
    return template.strip()


def build_caption_prompt(idea: ViralIdea) -> str:
    bullet_points = "\n".join(f"- {point}" for point in idea.key_points)
    template = f"""
You are writing a viral Instagram caption. Use the following idea details and produce a conversational caption.

Include:
- Hook (2 sentences max)
- Short narrative referencing the news
- 1 actionable takeaway
- Call-to-action inviting comments
- 4-6 relevant hashtags

Return plain text, no JSON.

Idea headline: {idea.headline}
Angle: {idea.angle}
Key points:
{bullet_points}
Source URL: {idea.source_url}
"""
    return template.strip()


