#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical Consultation Assistant - Web Crawler
Crawl disease information from medical websites (haodf.com, xywy.com)
Follows robots.txt compliance and crawler etiquette
"""

import os
import sys
import json
import time
import random
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Optional
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ===== Configuration =====

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Output file
OUTPUT_FILE = DATA_RAW_DIR / "diseases.json"
CHECKPOINT_FILE = DATA_RAW_DIR / ".crawl_checkpoint.json"

# Crawler settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15
DELAY_MIN = 3.0  # Minimum delay between requests (seconds)
DELAY_MAX = 5.0  # Maximum delay between requests (seconds)

# Random User-Agent pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ===== Logging Setup =====

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "crawler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ===== Robots.txt Checker =====

class RobotsChecker:
    """Check if a URL is allowed by robots.txt"""

    def __init__(self):
        self.parsers = {}

    def get_parser(self, base_url: str) -> RobotFileParser:
        """Get or create a robots.txt parser for a domain"""
        if base_url not in self.parsers:
            rp = RobotFileParser()
            robots_url = urljoin(base_url, "/robots.txt")
            rp.set_url(robots_url)
            try:
                rp.read()
                logger.info(f"[OK] Loaded robots.txt from {robots_url}")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load robots.txt from {robots_url}: {e}")
                # If we can't read robots.txt, allow everything (be conservative)
                rp.allow_all = True
            self.parsers[base_url] = rp
        return self.parsers[base_url]

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Check if a URL is allowed by robots.txt"""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        rp = self.get_parser(base_url)

        if hasattr(rp, "allow_all"):
            return True

        return rp.can_fetch(user_agent, url)

    def get_crawl_delay(self, base_url: str, user_agent: str = "*") -> Optional[float]:
        """Get the crawl delay specified in robots.txt"""
        rp = self.get_parser(base_url)
        try:
            delay = rp.crawl_delay(user_agent)
            return float(delay) if delay else None
        except Exception:
            return None


# ===== Crawler Class =====

class MedicalCrawler:
    """Crawl disease information from medical websites"""

    def __init__(self):
        self.session = requests.Session()
        self.robots_checker = RobotsChecker()
        self.crawled_urls = set()
        self.diseases = []
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        self._load_checkpoint()

    def _get_headers(self) -> dict:
        """Get random headers for each request"""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    def _delay(self):
        """Wait a random delay between requests"""
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
    )
    def _fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with retry logic"""
        # Check robots.txt first
        if not self.robots_checker.is_allowed(url):
            logger.warning(f"[BLOCKED] robots.txt disallows: {url}")
            return None

        logger.info(f"[FETCH] {url}")
        response = self.session.get(
            url,
            headers=self._get_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text

    def _load_checkpoint(self):
        """Load checkpoint to resume crawling"""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.crawled_urls = set(data.get("crawled_urls", []))
                self.diseases = data.get("diseases", [])
                self.stats = data.get("stats", self.stats)
                logger.info(
                    f"[RESUME] Loaded checkpoint: {len(self.crawled_urls)} URLs crawled, "
                    f"{len(self.diseases)} diseases collected"
                )
            except Exception as e:
                logger.warning(f"[WARN] Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save checkpoint for resume"""
        try:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "crawled_urls": list(self.crawled_urls),
                        "diseases": self.diseases,
                        "stats": self.stats,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"[ERROR] Failed to save checkpoint: {e}")

    def _save_results(self):
        """Save final results to JSON"""
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "source": "haodf.com + xywy.com",
                            "crawled_at": datetime.now().isoformat(),
                            "total_diseases": len(self.diseases),
                        },
                        "diseases": self.diseases,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"[SAVED] {len(self.diseases)} diseases saved to {OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")

    # ===== Haodf.com Crawlers =====

    def crawl_haodf_list(self) -> list:
        """Crawl disease list from haodf.com

        Note: The list page only shows ~15 popular diseases.
        For a complete list, we would need to crawl the sitemap
        or use a different approach (e.g., department-based crawling).
        """
        logger.info("=" * 60)
        logger.info("[START] Crawling haodf.com disease list")
        logger.info("=" * 60)

        disease_urls = []
        list_url = "https://www.haodf.com/jibing/"

        try:
            html = self._fetch(list_url)
            if not html:
                return disease_urls

            soup = BeautifulSoup(html, "html.parser")

            # Find all disease links with pattern: jibing-<disease_id>
            # These appear as: //www.haodf.com/citiao/jibing-gaoxueya/tuijian-doctor.html
            import re
            all_hrefs = [a.get("href", "") for a in soup.find_all("a", href=True)]

            for href in all_hrefs:
                # Match pattern: jibing-<disease_id>/
                match = re.search(r'jibing-([^/]+)', href)
                if match:
                    disease_id = match.group(1)
                    # Construct the detail URL
                    detail_url = f"https://www.haodf.com/citiao/jibing-{disease_id}.html"
                    if detail_url not in self.crawled_urls and detail_url not in disease_urls:
                        disease_urls.append(detail_url)

            logger.info(f"[LIST] Found {len(disease_urls)} disease URLs from haodf.com")
            logger.info(f"[INFO] Note: haodf.com list page only shows ~15 popular diseases")

        except Exception as e:
            logger.error(f"[ERROR] Failed to crawl disease list: {e}")

        return disease_urls

    def crawl_haodf_detail(self, url: str) -> Optional[dict]:
        """Crawl disease detail from haodf.com

        URL format: https://www.haodf.com/citiao/jibing-<disease_id>.html
        """
        try:
            html = self._fetch(url)
            if not html:
                return None

            soup = BeautifulSoup(html, "html.parser")
            disease = {"source": "haodf.com", "url": url, "crawled_at": datetime.now().isoformat()}

            # Extract disease name from H1
            h1_tag = soup.find("h1")
            if h1_tag:
                disease["name"] = h1_tag.get_text(strip=True)
            else:
                # Fallback to title
                title_tag = soup.find("title")
                if title_tag:
                    # Title format: "高血压 - 好大夫在线"
                    title_text = title_tag.get_text(strip=True)
                    disease["name"] = title_text.split(" - ")[0].strip() if " - " in title_text else title_text

            # Extract department from page content
            # Look for department info in the page
            page_text = soup.get_text()
            import re

            # Try to find department info (e.g., "心血管内科")
            dept_patterns = [
                r'([^\s]+科)',  # Match patterns like "心血管内科"
                r'就诊科室[：:]\s*([^\n]+)',
                r'科室[：:]\s*([^\n]+)',
            ]
            for pattern in dept_patterns:
                match = re.search(pattern, page_text)
                if match:
                    disease["department"] = match.group(1).strip()
                    break

            # Extract description from main content area
            # Look for sections with medical information
            content_sections = soup.find_all(["div", "section", "p"], class_=True)
            description_parts = []
            for section in content_sections[:5]:
                text = section.get_text(strip=True)
                if len(text) > 50 and len(text) < 500:  # Reasonable length for description
                    description_parts.append(text)

            if description_parts:
                disease["description"] = " ".join(description_parts)[:500]

            # Extract symptoms
            symptoms = []
            symptom_patterns = [
                r'症状[：:]\s*([^\n]+)',
                r'常见症状[：:]\s*([^\n]+)',
            ]
            for pattern in symptom_patterns:
                match = re.search(pattern, page_text)
                if match:
                    symptom_text = match.group(1)
                    # Split by common delimiters
                    symptoms = [s.strip() for s in re.split(r'[、，,]', symptom_text) if s.strip()]
                    break

            disease["symptoms"] = symptoms[:10]  # Limit to 10 symptoms

            # Extract treatment info
            treatment_patterns = [
                r'治疗[：:]\s*([^\n]+)',
                r'治疗方法[：:]\s*([^\n]+)',
            ]
            for pattern in treatment_patterns:
                match = re.search(pattern, page_text)
                if match:
                    disease["treatment"] = match.group(1).strip()[:300]
                    break

            return disease

        except Exception as e:
            logger.error(f"[ERROR] Failed to crawl {url}: {e}")
            return None

    # ===== XYWY.com Crawlers (Backup) =====

    def crawl_xywy_list(self) -> list:
        """Crawl disease list from xywy.com as backup source"""
        logger.info("=" * 60)
        logger.info("[START] Crawling xywy.com disease list (backup source)")
        logger.info("=" * 60)

        disease_urls = []
        list_url = "https://jib.xywy.com/"

        try:
            html = self._fetch(list_url)
            if not html:
                return disease_urls

            soup = BeautifulSoup(html, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "jib.xywy.com" in href and href.endswith(".htm"):
                    if href not in self.crawled_urls:
                        disease_urls.append(href)

            logger.info(f"[LIST] Found {len(disease_urls)} disease URLs from xywy.com")

        except Exception as e:
            logger.error(f"[ERROR] Failed to crawl xywy.com list: {e}")

        return disease_urls

    def crawl_xywy_detail(self, url: str) -> Optional[dict]:
        """Crawl disease detail from xywy.com"""
        try:
            html = self._fetch(url)
            if not html:
                return None

            soup = BeautifulSoup(html, "html.parser")
            disease = {"source": "xywy.com", "url": url, "crawled_at": datetime.now().isoformat()}

            # Extract disease name
            name_tag = soup.find("h1")
            if name_tag:
                disease["name"] = name_tag.get_text(strip=True)

            # Extract description
            desc_div = soup.find("div", class_="jib-jj") or soup.find("div", class_="content")
            if desc_div:
                disease["description"] = desc_div.get_text(strip=True)[:500]

            # Extract symptoms, department, etc.
            for label in ["症状", "科室", "治疗", "药物"]:
                tag = soup.find(string=lambda s: s and label in s if s else False)
                if tag:
                    parent = tag.find_parent()
                    if parent:
                        text = parent.get_text(strip=True)
                        if label == "症状":
                            disease["symptoms"] = [s.strip() for s in text.split("、") if s.strip()]
                        elif label == "科室":
                            disease["department"] = text
                        elif label == "治疗":
                            disease["treatment"] = text[:300]

            return disease

        except Exception as e:
            logger.error(f"[ERROR] Failed to crawl {url}: {e}")
            return None

    # ===== Main Crawl Logic =====

    def run(self, max_diseases: int = 100, use_backup: bool = False):
        """
        Run the crawler

        Args:
            max_diseases: Maximum number of diseases to crawl
            use_backup: Whether to use xywy.com as backup source
        """
        logger.info("=" * 60)
        logger.info(f"[START] Medical Crawler started at {datetime.now().isoformat()}")
        logger.info(f"[CONFIG] Max diseases: {max_diseases}, Backup source: {use_backup}")
        logger.info("=" * 60)

        # Step 1: Crawl from haodf.com (primary source)
        haodf_urls = self.crawl_haodf_list()

        for url in haodf_urls:
            if len(self.diseases) >= max_diseases:
                break

            if url in self.crawled_urls:
                self.stats["skipped"] += 1
                continue

            disease = self.crawl_haodf_detail(url)
            if disease and disease.get("name"):
                self.diseases.append(disease)
                self.crawled_urls.add(url)
                self.stats["success"] += 1
                logger.info(
                    f"[OK] [{len(self.diseases)}/{max_diseases}] "
                    f"{disease['name']} - {len(disease.get('symptoms', []))} symptoms"
                )
            else:
                self.stats["failed"] += 1

            self.stats["total"] += 1
            self._save_checkpoint()
            self._delay()

        # Step 2: Crawl from xywy.com (backup source)
        if use_backup and len(self.diseases) < max_diseases:
            xywy_urls = self.crawl_xywy_list()

            for url in xywy_urls:
                if len(self.diseases) >= max_diseases:
                    break

                if url in self.crawled_urls:
                    self.stats["skipped"] += 1
                    continue

                disease = self.crawl_xywy_detail(url)
                if disease and disease.get("name"):
                    self.diseases.append(disease)
                    self.crawled_urls.add(url)
                    self.stats["success"] += 1
                    logger.info(
                        f"[OK] [{len(self.diseases)}/{max_diseases}] "
                        f"{disease['name']} (xywy)"
                    )
                else:
                    self.stats["failed"] += 1

                self.stats["total"] += 1
                self._save_checkpoint()
                self._delay()

        # Step 3: Save results
        self._save_results()

        # Step 4: Print summary
        logger.info("=" * 60)
        logger.info("[COMPLETE] Crawl Summary:")
        logger.info(f"  Total requests: {self.stats['total']}")
        logger.info(f"  Success: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Skipped (already crawled): {self.stats['skipped']}")
        logger.info(f"  Diseases collected: {len(self.diseases)}")
        logger.info(f"  Output: {OUTPUT_FILE}")
        logger.info("=" * 60)


# ===== Main Entry =====

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Medical Website Crawler")
    parser.add_argument(
        "--max", type=int, default=100, help="Maximum number of diseases to crawl (default: 100)"
    )
    parser.add_argument(
        "--backup", action="store_true", help="Use xywy.com as backup source"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset checkpoint and start fresh"
    )

    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("[RESET] Checkpoint file deleted, starting fresh")

    crawler = MedicalCrawler()
    crawler.run(max_diseases=args.max, use_backup=args.backup)


if __name__ == "__main__":
    main()
