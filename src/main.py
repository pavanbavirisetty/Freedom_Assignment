from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from src.images.generator import CreativeEngine
from src.llm.idea_generator import IdeaGenerator
from src.news.service import NewsService
from src.utils.config import InstagramAsset, get_settings
from src.utils.logger import get_logger


@contextmanager
def _temporary_settings_override(**overrides: str) -> Generator[None, None, None]:
    original_values: dict[str, str | None] = {}
    for key, value in overrides.items():
        env_key = key.upper()
        original_values[env_key] = os.environ.get(env_key)
        os.environ[env_key] = value

    get_settings.cache_clear()
    try:
        yield
    finally:
        for env_key, original in original_values.items():
            if original is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original
        get_settings.cache_clear()


@contextmanager
def _noop_context() -> Generator[None, None, None]:
    yield


def run_pipeline(industry: str | None = None) -> list[InstagramAsset]:
    context = _temporary_settings_override(industry=industry) if industry else _noop_context()
    with context:
        settings = get_settings()
        logger = get_logger("Pipeline")

        logger.info("Starting pipeline for industry '%s'", settings.industry)
        news_service = NewsService(settings=settings)
        idea_generator = IdeaGenerator()
        creative_engine = CreativeEngine()

        articles = news_service.collect_articles()
        if not articles:
            logger.warning("No articles available; aborting.")
            return []

        ideas = idea_generator.generate_ideas(articles)
        if not ideas:
            logger.warning("No viral ideas generated; aborting.")
            return []

        assets: list[InstagramAsset] = []
        timestamp_dir = settings.output_dir / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        for idea in ideas:
            try:
                caption = idea_generator.generate_caption(idea)
                image_result = creative_engine.create_asset(idea, output_dir=timestamp_dir)
                asset = InstagramAsset(idea=idea, caption=caption, image_path=image_result.image_path)
                assets.append(asset)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to create asset for idea '%s': %s", idea.headline, exc)

        if not assets:
            logger.warning("No assets created.")
            return []

        metadata = [
            {
                "headline": asset.idea.headline,
                "angle": asset.idea.angle,
                "key_points": asset.idea.key_points,
                "source_url": asset.idea.source_url,
                "caption": asset.caption,
                "image_path": str(asset.image_path),
            }
            for asset in assets
        ]
        metadata_path = timestamp_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        logger.info("Pipeline complete. Assets saved to %s", timestamp_dir)
        return assets


if __name__ == "__main__":
    run_pipeline()


