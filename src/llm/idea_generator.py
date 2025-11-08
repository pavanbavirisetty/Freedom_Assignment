from __future__ import annotations

import json
import re
from typing import List, Optional

from src.llm.ollama_client import (
    OllamaClient,
    build_caption_prompt,
    build_viral_idea_prompt,
)
from src.utils.config import InstagramAsset, NewsArticle, ViralIdea, get_settings
from src.utils.logger import get_logger


class IdeaGenerator:
    """Generates viral ideas and captions using Ollama."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.client = client or OllamaClient()

    def generate_ideas(self, articles: List[NewsArticle]) -> List[ViralIdea]:
        ideas: List[ViralIdea] = []
        for article in articles:
            prompt = build_viral_idea_prompt(article)
            try:
                response = self.client.generate(prompt, temperature=0.3)
                json_payload = self._extract_json_object(response)
                if not json_payload:
                    self.logger.warning("LLM response missing JSON for article %s. Applying heuristic parse.", article.url)
                    idea = self._heuristic_parse(article, response)
                    if idea:
                        ideas.append(idea)
                    continue
                data = json.loads(json_payload)
                idea = ViralIdea(
                    headline=data["headline"],
                    angle=data["angle"],
                    key_points=data["key_points"],
                    source_url=article.url,
                )
                ideas.append(idea)
            except json.JSONDecodeError:
                self.logger.exception("Failed to parse idea JSON for article %s", article.url)
            except KeyError:
                self.logger.exception("Missing fields in idea JSON for article %s", article.url)
        self.logger.info("Generated %d viral ideas", len(ideas))
        return ideas

    def generate_caption(self, idea: ViralIdea) -> str:
        prompt = build_caption_prompt(idea)
        caption = self.client.generate(prompt, temperature=0.6)
        return caption.strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        if not text:
            return None

        # remove markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            if text.endswith("```"):
                text = text[:-3].strip()

        # find first JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0).strip() if match else None

    def _heuristic_parse(self, article: NewsArticle, response: str) -> Optional[ViralIdea]:
        headline: Optional[str] = None
        angle: Optional[str] = None
        key_points: List[str] = []

        collecting_points = False
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line:
                if collecting_points and key_points:
                    collecting_points = False
                continue

            normalized = line.lower()

            if not headline and "headline" in normalized:
                headline = line.split(":", 1)[-1].strip(" -*_\"'")
                continue

            if not angle and "angle" in normalized:
                angle = line.split(":", 1)[-1].strip(" -*_\"'")
                continue

            if "key points" in normalized or "slides" in normalized:
                collecting_points = True
                continue

            if collecting_points:
                cleaned = line.lstrip("-*1234567890.() ").strip()
                if cleaned:
                    key_points.append(cleaned)

        if not headline:
            headline = article.title
        if not angle:
            angle = article.summary or "Fresh perspective on current news."
        if not key_points:
            key_points = [
                "Summarize the core news story and why it matters.",
                "Highlight the key people or stakes involved.",
                "Explain what audiences should watch for next.",
            ]

        try:
            return ViralIdea(
                headline=headline,
                angle=angle,
                key_points=key_points,
                source_url=article.url,
            )
        except Exception:  # pylint: disable=broad-except
            self.logger.error("Heuristic parse failed for article %s with response: %s", article.url, response)
            return None


