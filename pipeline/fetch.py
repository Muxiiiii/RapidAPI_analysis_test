"""Step 0 (optional): download job postings from the Daily International Job Postings API on RapidAPI.

    export RAPIDAPI_KEY=...            # never put the key in a file that is committed
    python -m pipeline.fetch --date-min 2026-04-01 --date-max 2026-05-01 --out raw_2026-04.csv
    python run_pipeline.py --input combined_jobs_data.csv raw_2026-04.csv

Each page is one API request and counts against the RapidAPI plan's quota, so --max-pages caps a run.
Use the same filters for every batch: batches collected with different filters cannot be compared over time."""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

HOST = 'daily-international-job-postings.p.rapidapi.com'
URL = f'https://{HOST}/api/v2/jobs/search'
# Standard filters. Keep these fixed between batches so the batches stay comparable.
DEFAULT_PARAMS = {'countryCode': 'us', 'format': 'json', 'hasSalary': 'true'}


class FetchError(Exception):
    pass


def get_page(session, key, params, page, retries=4, timeout=30):
    """One page of results. Retries on rate limits (429) and server errors with a growing wait."""
    for attempt in range(retries + 1):
        r = session.get(URL, headers={'x-rapidapi-host': HOST, 'x-rapidapi-key': key},
                        params={**params, 'page': page}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise FetchError(f'The API rejected the key (HTTP {r.status_code}). Check RAPIDAPI_KEY and the subscription.')
        if r.status_code == 429 or r.status_code >= 500:
            if attempt < retries:
                wait = 2 ** (attempt + 1)
                log.warning('HTTP %s on page %s, retrying in %ss', r.status_code, page, wait)
                time.sleep(wait)
                continue
        raise FetchError(f'HTTP {r.status_code} on page {page}: {r.text[:200]}')


def fetch(key, date_min, date_max, max_pages=30, start_page=1, pause=1.0, extra_params=None, session=None):
    """Download postings created between date_min and date_max (YYYY-MM-DD). Returns a list of dicts."""
    session = session or requests.Session()
    params = {**DEFAULT_PARAMS, 'dateCreatedMin': date_min, 'dateCreatedMax': date_max, **(extra_params or {})}
    jobs, page_size = [], None
    for page in range(start_page, start_page + max_pages):
        data = get_page(session, key, params, page)
        batch = data.get('result', [])
        if page == start_page:
            page_size = len(batch)
            log.info('API reports %s matching postings, %s per page', data.get('total', 'an unknown number of'), page_size)
        if not batch:
            log.info('Page %s is empty: no more results', page)
            break
        jobs.extend(batch)
        log.info('Page %s: %s postings (total %s)', page, len(batch), len(jobs))
        if len(batch) < page_size:
            break
        time.sleep(pause)
    return jobs


def to_frame(jobs):
    """Flatten the nested JSON the same way the original collection notebook did (pd.json_normalize)."""
    return pd.json_normalize(jobs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--date-min', required=True, help='earliest posting date, YYYY-MM-DD')
    p.add_argument('--date-max', required=True, help='latest posting date, YYYY-MM-DD')
    p.add_argument('--out', required=True, type=Path, help='CSV file to write')
    p.add_argument('--max-pages', type=int, default=30, help='maximum number of API requests (default 30)')
    p.add_argument('--start-page', type=int, default=1)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    key = os.environ.get('RAPIDAPI_KEY')
    if not key:
        log.error('RAPIDAPI_KEY is not set. Run:  export RAPIDAPI_KEY=your-key')
        return 1
    if args.out.exists():
        log.error('%s already exists. Choose another name so earlier data is not overwritten.', args.out)
        return 1
    try:
        jobs = fetch(key, args.date_min, args.date_max, args.max_pages, args.start_page)
    except FetchError as e:
        log.error('%s', e)
        return 1
    if not jobs:
        log.error('The API returned no postings for these dates. Nothing written.')
        return 1
    df = to_frame(jobs)
    df.to_csv(args.out, index=False)
    log.info('Wrote %s postings, %s columns to %s', len(df), df.shape[1], args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
