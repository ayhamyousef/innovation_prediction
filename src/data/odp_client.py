"""
USPTO Open Data Portal (ODP) API client.

Fetches granted patent metadata via the ODP patent file-wrapper search API.
Handles pagination, rate limiting, retries, and per-month caching.

API base: https://api.uspto.gov/api
Auth: X-API-KEY header (get key at https://data.uspto.gov)
"""

import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger("innovation_prediction.odp_client")

API_BASE = "https://api.uspto.gov/api/v1/patent/applications/search"
PAGE_SIZE = 100
MAX_RETRIES = 5

# Minimal field set to keep responses small
FIELDS = [
    "applicationMetaData.patentNumber",
    "applicationMetaData.inventionTitle",
    "applicationMetaData.grantDate",
    "applicationMetaData.filingDate",
    "applicationMetaData.cpcClassificationBag",
    "applicationMetaData.inventorBag",
    "applicationMetaData.applicationTypeCode",
]


class ODPClient:
    """
    Client for the USPTO ODP API. Fetches granted utility patents by month,
    caches results as JSONL, and returns records in our standard format.

    Note: This API provides CPC codes (successor to IPC) and patent titles.
    Abstracts are not available via this endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache",
                 delay_sec: float = 1.0, **kwargs):
        self.api_key = api_key or os.environ.get("ODP_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_sec = delay_sec
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        if not self.api_key:
            logger.warning("No ODP_API_KEY set. API calls will fail.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_patents_range(self, start_year: int, end_year: int,
                            **kwargs) -> List[Dict]:
        """Fetch granted utility patents for a range of years, month by month."""
        all_patents = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                patents = self._fetch_month(year, month)
                all_patents.extend(patents)
            logger.info(f"Total patents through {year}: {len(all_patents)}")
        return all_patents

    def fetch_patents_by_year(self, year: int, **kwargs) -> List[Dict]:
        """Fetch patents for a single year."""
        return self.fetch_patents_range(year, year)

    # ------------------------------------------------------------------
    # Per-month fetch with caching
    # ------------------------------------------------------------------
    def _fetch_month(self, year: int, month: int) -> List[Dict]:
        """Fetch all granted utility patents for a single month."""
        cache_file = self.cache_dir / f"patents_{year}_{month:02d}.jsonl"
        if cache_file.exists():
            records = self._read_jsonl(cache_file)
            logger.info(f"  {year}-{month:02d}: {len(records)} patents (cached)")
            return records

        # Date range for this month
        date_from = f"{year}-{month:02d}-01"
        if month == 12:
            date_to = f"{year}-12-31"
        else:
            date_to = f"{year}-{month + 1:02d}-01"
            # Subtract one day by using the last day of current month
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            date_to = f"{year}-{month:02d}-{last_day}"

        # First request to get total count
        data = self._api_request(date_from, date_to, offset=0)
        if data is None:
            logger.error(f"  {year}-{month:02d}: failed to fetch count")
            return []

        total = data.get("count", 0)
        logger.info(f"  {year}-{month:02d}: {total} utility patents to fetch")

        if total == 0:
            self._write_jsonl([], cache_file)
            return []

        # Collect first page
        all_raw = data.get("patentFileWrapperDataBag", [])

        # Paginate through remaining pages
        pages_needed = (total + PAGE_SIZE - 1) // PAGE_SIZE
        for page in tqdm(range(1, pages_needed), desc=f"{year}-{month:02d}",
                         disable=pages_needed <= 2):
            offset = page * PAGE_SIZE
            time.sleep(self.delay_sec)
            page_data = self._api_request(date_from, date_to, offset=offset)
            if page_data is None:
                logger.warning(f"  Failed at offset {offset}, stopping month")
                break
            batch = page_data.get("patentFileWrapperDataBag", [])
            if not batch:
                break
            all_raw.extend(batch)

        # Normalize records
        records = []
        for raw in all_raw:
            normalized = self._normalize_record(raw)
            if normalized:
                records.append(normalized)

        self._write_jsonl(records, cache_file)
        logger.info(f"  {year}-{month:02d}: cached {len(records)} patents")
        return records

    # ------------------------------------------------------------------
    # API request with retries
    # ------------------------------------------------------------------
    def _api_request(self, date_from: str, date_to: str,
                     offset: int = 0) -> Optional[Dict]:
        """Make a single search request with retry logic."""
        payload = {
            "q": "applicationMetaData.applicationTypeCode:UTL",
            "rangeFilters": [
                {
                    "field": "applicationMetaData.grantDate",
                    "valueFrom": date_from,
                    "valueTo": date_to,
                }
            ],
            "pagination": {"offset": offset, "limit": PAGE_SIZE},
            "sort": [{"field": "applicationMetaData.grantDate", "order": "asc"}],
            "fields": FIELDS,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(API_BASE, json=payload, timeout=60)

                if resp.status_code == 429:
                    wait = self.delay_sec * (2 ** attempt) + 5
                    logger.warning(f"Rate limited, waiting {wait:.0f}s "
                                   f"(attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                wait = self.delay_sec * (2 ** attempt)
                logger.warning(f"API request failed: {e} "
                               f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)

        return None

    # ------------------------------------------------------------------
    # Normalize to standard format
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_record(raw: Dict) -> Optional[Dict]:
        """Convert an ODP API record to our standard format."""
        try:
            meta = raw.get("applicationMetaData", {})
            patent_number = meta.get("patentNumber", "")
            if not patent_number:
                return None

            title = meta.get("inventionTitle", "") or ""
            grant_date = meta.get("grantDate", "") or ""
            filing_date = meta.get("filingDate", "") or ""
            grant_year = int(grant_date[:4]) if grant_date else None

            # CPC codes (successor to IPC — same hierarchical structure)
            cpc_codes_raw = meta.get("cpcClassificationBag", []) or []
            cpc_codes = [c.strip().replace(" ", "") for c in cpc_codes_raw if c]

            # Inventors
            inventor_bag = meta.get("inventorBag", []) or []
            inventor_count = len(inventor_bag)

            # Assignee info not directly in search results — use customer number as proxy
            assignee_names = []

            # Map CPC codes to ipc_codes fields for downstream compatibility
            # (CPC is the successor to IPC with the same hierarchical structure)
            codes_6digit = list(set(c[:6] for c in cpc_codes if len(c) >= 6))

            return {
                "patent_number": patent_number,
                "title": title,
                "abstract": "",  # not available from this API
                "patent_date": grant_date,
                "application_date": filing_date,
                "patent_year": grant_year,
                "cpc_codes": cpc_codes,
                "ipc_codes": cpc_codes,             # alias for downstream compatibility
                "ipc_codes_6digit": codes_6digit,    # alias for downstream compatibility
                "inventor_count": inventor_count,
                "assignee_names": assignee_names,
                "assignee_count": len(assignee_names),
                "num_claims": 0,  # not in search fields
            }
        except Exception as e:
            logger.debug(f"Failed to normalize record: {e}")
            return None

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _write_jsonl(records: List[Dict], path: Path):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
