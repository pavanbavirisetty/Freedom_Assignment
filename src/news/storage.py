from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src.utils.config import NewsArticle, get_settings
from src.utils.logger import get_logger


metadata = MetaData()

articles_table = Table(
    "articles",
    metadata,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("url", String, nullable=False, unique=True),
    Column("source", String, nullable=True),
    Column("summary", String, nullable=True),
    Column("published_at", String, nullable=True),
    Column("industry", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    UniqueConstraint("url", name="uniq_article_url"),
)


class NewsRepository:
    """Handles persistence of scraped articles."""

    def __init__(self, engine: Optional[Engine] = None) -> None:
        settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.engine = engine or create_engine(f"sqlite:///{settings.sqlite_db_path}", future=True)
        metadata.create_all(self.engine)

    @contextmanager
    def _connect(self):
        with self.engine.begin() as conn:
            yield conn

    def upsert_articles(self, industry: str, articles: Iterable[NewsArticle]) -> int:
        """Insert articles, skipping duplicates by URL."""

        inserted = 0
        with self._connect() as conn:
            for article in articles:
                stmt = insert(articles_table).values(
                    id=self._make_id(article.url),
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    summary=article.summary,
                    published_at=article.published_at,
                    industry=industry,
                    created_at=datetime.utcnow(),
                )
                try:
                    conn.execute(stmt)
                    inserted += 1
                except IntegrityError:
                    self.logger.debug("Article already exists: %s", article.url)
        self.logger.info("Stored %d new articles", inserted)
        return inserted

    def fetch_latest(self, industry: str, limit: int = 10) -> List[NewsArticle]:
        with self._connect() as conn:
            stmt = (
                select(
                    articles_table.c.title,
                    articles_table.c.url,
                    articles_table.c.source,
                    articles_table.c.summary,
                    articles_table.c.published_at,
                )
                .where(articles_table.c.industry == industry)
                .order_by(articles_table.c.created_at.desc())
                .limit(limit)
            )
            result = conn.execute(stmt)
            rows = result.fetchall()

        return [
            NewsArticle(
                title=row.title,
                url=row.url,
                source=row.source,
                summary=row.summary,
                published_at=row.published_at,
            )
            for row in rows
        ]

    @staticmethod
    def _make_id(url: str) -> str:
        return url


