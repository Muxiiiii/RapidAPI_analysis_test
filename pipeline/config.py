"""Paths and thresholds for the pipeline. Change them here, not inside the steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_INPUTS = [ROOT / 'combined_jobs_data.csv']   # default raw export(s) from the job postings API
CLEAN_CSV = ROOT / 'jobs_clean.csv'              # read by the Streamlit app (app.py)
OUTPUT_DIR = ROOT / 'outputs'                    # tables and the run report

# Columns the raw export must contain for the pipeline to run
REQUIRED_COLUMNS = ['title', 'company', 'city', 'state', 'source', 'industry', 'skills', 'minSalary',
                    'dateCreated', 'dateActive', 'careerLevel', 'workPlace', 'workType', 'contractType',
                    'department', 'isDirect', 'isRecruiter']
# Columns carried through cleaning. A direct API download has about 85 columns (nested jsonLD fields);
# only these are kept so the cleaned file stays the same shape whatever the source of the raw file.
RAW_COLUMNS = ['occupation', 'dateActive', 'city', 'timezone', 'contractType', 'language', 'industry', 'source',
               'locale', 'title', 'skills', 'dateCreated', 'timezoneOffset', 'countryCode', 'company', 'isDuplicate',
               'portal', 'department', 'workPlace', 'isRecruiter', 'minSalary', 'hasSalary', 'careerLevel',
               'workType', 'postCode', 'isDirect', 'geoPoint.lon', 'geoPoint.lat', 'state', 'dateExpired']
# Dropped when present: constant, duplicated or almost entirely missing in the API export
DROP_COLUMNS = ['language', 'locale', 'countryCode', 'hasSalary', 'postCode', 'portal', 'dateExpired',
                'timezone', 'timezoneOffset', 'geoPoint.lon', 'geoPoint.lat']
LIST_COLUMNS = ['skills', 'workPlace', 'careerLevel', 'workType', 'contractType']
DEDUP_KEYS = ['title', 'company', 'city', 'state', 'minSalary']   # same posting scraped more than once

# Salary unit inference (the API gives no unit)
HOURLY_MIN, HOURLY_MAX = 7.25, 100     # federal minimum wage up to 100 = hourly pay
ANNUAL_MIN = 20000                     # 20,000 and above = annual pay
HOURS_PER_YEAR = 2080

# Data quality limits: the run fails when these are broken
MIN_ROWS = 500                 # fewer cleaned postings than this means the input is probably wrong
MAX_UNKNOWN_SALARY = 0.05      # share of postings whose salary unit cannot be inferred
MIN_WITH_SKILLS = 0.30         # share of postings with at least one curated skill
