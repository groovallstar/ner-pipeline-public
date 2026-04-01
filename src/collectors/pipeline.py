import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from collectors.ollama_ner_labeler import OllamaNERLabeler
from collectors.web_crawler import WebCrawler

logger = logging.getLogger(__name__)


class NERPipeline:
    def __init__(
        self,
        crawler: WebCrawler,
        labeler: OllamaNERLabeler,
        output_path: Optional[str] = None,
    ) -> None:
        self.crawler = crawler
        self.labeler = labeler
        self.output_path = output_path

    def run(self, urls: List[str]) -> List[dict]:
        """Crawl URLs, label with NER, optionally save to JSONL. Returns NERRecord list."""
        if not urls:
            return []

        t0 = time.time()
        raw_records = self.crawler.crawl_urls(urls)
        t1 = time.time()
        print(f"[TIMER] crawl: {t1-t0:.2f}s  ({len(raw_records)}/{len(urls)} URLs)")
        logger.info("Crawled %d/%d URLs successfully", len(raw_records), len(urls))

        records = self.labeler.label_records(raw_records)
        t2 = time.time()
        print(f"[TIMER] label: {t2-t1:.2f}s  ({len(records)} records)")
        logger.info("Labeled %d NER records", len(records))

        if self.output_path and records:
            self._save_jsonl(records, self.output_path)
            t3 = time.time()
            print(f"[TIMER] save:  {t3-t2:.2f}s")

        return records

    def _save_jsonl(self, records: List[dict], path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d records → %s", len(records), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run NER pipeline: crawl + label + save")
    parser.add_argument("--urls", nargs="+", required=True, help="URLs to crawl")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--model", default="qwen3.5:27b", help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Context window size (smaller = faster)")
    parser.add_argument("--batch-size", type=int, default=10, help="Sentences per LLM call (batch processing)")
    args = parser.parse_args()

    pipeline = NERPipeline(
        crawler=WebCrawler(),
        labeler=OllamaNERLabeler(model=args.model, base_url=args.base_url, num_ctx=args.num_ctx, batch_size=args.batch_size),
        output_path=args.output,
    )
    records = pipeline.run(args.urls)
    print(f"Done. {len(records)} NER records saved to {args.output}")
