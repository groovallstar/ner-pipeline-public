import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# NOTE: Callers are responsible for ensuring target URLs permit automated access.
# Check robots.txt and terms of service before crawling any site.

logger = logging.getLogger(__name__)


class RawTextRecord(dict):
    """Collected raw text record: text, source_url, collected_at (ISO 8601)."""
    text: str
    source_url: str
    collected_at: str


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; NERPipelineCrawler/1.0)"
)


class WebCrawler:
    def __init__(
        self,
        delay_seconds: float = 1.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        timeout: int = 10,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def crawl_urls(self, urls: List[str]) -> List[dict]:
        """Fetch plain text from a list of URLs. Skips unreachable URLs."""
        records = []
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(self.delay_seconds)
            record = self._fetch_text(url)
            if record is not None:
                records.append(record)
        return records

    def _fetch_text(self, url: str) -> Optional[dict]:
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning("Skipping %s: %s", url, e)
            return None

        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception:
            soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)
        return {
            "text": text,
            "source_url": url,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

    def crawl_api(
        self,
        url: str,
        text_path: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Fetch records from a JSON REST API. Returns a list with one record on success."""
        try:
            response = self._session.get(
                url, headers=headers or {}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("API request failed for %s: %s", url, e)
            return []

        try:
            text = str(self._extract_json_path(data, text_path))
        except KeyError as e:
            logger.warning("JSON path '%s' not found in response from %s: %s", text_path, url, e)
            return []

        return [{
            "text": text,
            "source_url": url,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }]

    @staticmethod
    def _extract_json_path(data: dict, path: str):
        """Navigate nested dict using dot notation (e.g. 'data.items.text')."""
        for key in path.split("."):
            data = data[key]
        return data
