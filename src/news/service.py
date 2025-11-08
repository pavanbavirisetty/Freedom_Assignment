from __future__ import annotations

from typing import Iterable, List

from src.news.scraper import NewsScraper, load_scrapers
from src.news.storage import NewsRepository
from src.utils.config import NewsArticle, Settings, get_settings
from src.utils.logger import get_logger


class NewsService:
    """Coordinates scraping and persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        repository: NewsRepository | None = None,
        scrapers: Iterable[NewsScraper] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.repository = repository or NewsRepository()
        self.scrapers = list(
            scrapers
            or load_scrapers(
                industry=self.settings.industry,
                max_articles=self.settings.max_articles,
                names=self.settings.news_sources,
            )
        )

    def collect_articles(self) -> List[NewsArticle]:
        all_articles: List[NewsArticle] = []
        for scraper in self.scrapers:
            try:
                articles = scraper.fetch()
                all_articles.extend(articles)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Error fetching from %s: %s", scraper.__class__.__name__, exc)

        if not all_articles:
            self.logger.warning("No articles fetched for industry '%s'", self.settings.industry)
            return []

        self.repository.upsert_articles(self.settings.industry, all_articles)
        latest = self.repository.fetch_latest(self.settings.industry, self.settings.max_articles)
        self.logger.info("Returning %d articles", len(latest))
        return latest


