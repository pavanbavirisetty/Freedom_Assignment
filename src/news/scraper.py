from __future__ import annotations

import datetime as dt
from typing import Iterable, List
from urllib.parse import quote_plus

import feedparser
import requests

from src.utils.config import NewsArticle
from src.utils.logger import get_logger


class NewsScraper:
    """Base class for news scrapers."""

    def __init__(self, industry: str, max_articles: int = 10) -> None:
        self.industry = industry
        self.max_articles = max_articles
        self.logger = get_logger(self.__class__.__name__)

    def fetch(self) -> List[NewsArticle]:
        """Fetch article data."""

        articles = list(self._fetch_impl())
        self.logger.info("Fetched %d articles", len(articles))
        return articles[: self.max_articles]

    def _fetch_impl(self) -> Iterable[NewsArticle]:
        raise NotImplementedError


class GoogleNewsScraper(NewsScraper):
    """Scrape top stories from Google News search results."""

    GOOGLE_NEWS_URL = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    def _fetch_impl(self) -> Iterable[NewsArticle]:
        query = quote_plus(self.industry)
        url = self.GOOGLE_NEWS_URL.format(query=query)
        self.logger.info("Requesting Google News feed for '%s'", self.industry)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        self.logger.debug("Feed contains %d entries", len(feed.entries))

        for entry in feed.entries:
            title = entry.get("title")
            link = entry.get("link")
            if not title or not link:
                continue

            summary = entry.get("summary") or entry.get("description")
            published = entry.get("published") or entry.get("updated")
            source = entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else None

            yield NewsArticle(
                title=title,
                url=link,
                source=source or "Google News",
                summary=summary,
                published_at=published,
            )


SCRAPER_REGISTRY = {
    "google_news": GoogleNewsScraper,
}


def load_scrapers(industry: str, max_articles: int, names: Iterable[str]) -> List[NewsScraper]:
    scrapers: List[NewsScraper] = []
    for name in names:
        scraper_cls = SCRAPER_REGISTRY.get(name)
        if not scraper_cls:
            raise KeyError(f"Unsupported news source '{name}'. Available: {list(SCRAPER_REGISTRY)}")
        scrapers.append(scraper_cls(industry=industry, max_articles=max_articles))
    return scrapers


