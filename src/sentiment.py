"""
Module: sentiment.py
Responsibility: Fetch pre-market & intraday news headlines via Google News RSS and score sentiment.
Acts as a Gatekeeper / Safety Veto to prevent buying on negative overnight news.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import re
from typing import List, Dict, Any, Optional
import urllib.parse
import feedparser

logger = logging.getLogger(__name__)

# Core financial sentiment lexicon for zero-cost, offline rule-based scoring
BULLISH_KEYWORDS = {
    "surge", "soar", "gain", "profit", "jump", "rally", "record", "growth",
    "upgrade", "bullish", "beat", "positive", "high", "order", "contract",
    "expansion", "boost", "strong", "outperform", "dividend", "acquisition"
}

BEARISH_KEYWORDS = {
    "fall", "drop", "plunge", "loss", "crash", "slump", "probe", "investigation",
    "penalty", "fine", "fraud", "downgrade", "bearish", "miss", "weak", "strike",
    "halt", "cut", "debt", "default", "warning", "ban", "resignation", "scam"
}


@dataclass
class NewsItem:
    title: str
    published: str
    link: str
    score: float


@dataclass
class SentimentResult:
    query: str
    overall_score: float         # Normalized -1.0 to +1.0
    headline_count: int
    is_veto_active: bool         # True if negative sentiment blocks trades
    veto_reason: Optional[str]
    headlines: List[NewsItem] = field(default_factory=list)


class NewsSentimentEngine:
    def __init__(self, config: Dict[str, Any]):
        sentiment_cfg = config.get("sentiment", {})
        self.query = sentiment_cfg.get("query", "Tata Motors")
        self.veto_threshold = float(sentiment_cfg.get("veto_threshold", -0.30))

    def _build_rss_url(self, lookback_window: str = "1d") -> str:
        """
        Builds Google News RSS URL filtered by Indian region and time window (e.g. '1d', '1h', '7d').
        """
        encoded_query = urllib.parse.quote(f"{self.query} when:{lookback_window}")
        return f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    def _score_headline_lexicon(self, text: str) -> float:
        """
        Fast, robust keyword-based sentiment scoring (-1.0 to +1.0).
        Works 100% offline with zero dependencies.
        """
        tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos_hits = sum(1 for w in tokens if w in BULLISH_KEYWORDS)
        neg_hits = sum(1 for w in tokens if w in BEARISH_KEYWORDS)

        total = pos_hits + neg_hits
        if total == 0:
            return 0.0
        return (pos_hits - neg_hits) / total

    def fetch_news_sentiment(self, lookback_window: str = "1d") -> SentimentResult:
        """
        Fetches headlines from Google News RSS and computes aggregate sentiment score.
        """
        rss_url = self._build_rss_url(lookback_window)
        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            logger.error(f"Error parsing Google News RSS feed: {e}")
            return SentimentResult(
                query=self.query,
                overall_score=0.0,
                headline_count=0,
                is_veto_active=False,
                veto_reason=f"News feed unreachable ({e}); defaulting to neutral",
                headlines=[]
            )

        items: List[NewsItem] = []
        scores: List[float] = []

        for entry in feed.entries[:15]:  # Top 15 most relevant recent items
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "")
            score = self._score_headline_lexicon(title)
            scores.append(score)
            items.append(NewsItem(title=title, published=published, link=link, score=score))

        if not scores:
            overall_score = 0.0
        else:
            # Weighted average favoring headlines with stronger sentiment
            overall_score = sum(scores) / len(scores)
            overall_score = max(-1.0, min(1.0, round(overall_score, 2)))

        is_veto = overall_score <= self.veto_threshold
        veto_reason = None
        if is_veto:
            veto_reason = f"Pre-market sentiment ({overall_score}) is below veto threshold ({self.veto_threshold})"

        return SentimentResult(
            query=self.query,
            overall_score=overall_score,
            headline_count=len(items),
            is_veto_active=is_veto,
            veto_reason=veto_reason,
            headlines=items
        )
