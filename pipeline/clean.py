"""Step 1: load the raw API export and clean it. Same rules as jobs_analysis.ipynb, section 2."""
import ast
import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

PLACEHOLDER = {'N/A', 'None', 'Not Specified'}
LEVEL = {'Intern': 0, 'Student': 0, 'Apprentice': 0, 'Junior': 1, 'Medior': 2, 'Senior': 3, 'Lead': 4,
         'Director': 5, 'Chief': 6}
LABEL = {0: 'Intern', 1: 'Junior', 2: 'Medior', 3: 'Senior', 4: 'Lead', 5: 'Director', 6: 'Chief'}
US_STATES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'New York City': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}


def load_raw(paths):
    """Read one or more raw CSV exports and stack them."""
    frames = []
    for p in paths:
        f = pd.read_csv(p, low_memory=False)
        missing = [c for c in config.REQUIRED_COLUMNS if c not in f.columns]
        if missing:
            raise ValueError(f'{p} is missing required columns: {missing}')
        frames.append(f[[c for c in config.RAW_COLUMNS if c in f.columns]])
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def parse_list(x):
    """"['a', 'N/A']" -> ['a']. The API stores lists as strings and uses placeholders for empty."""
    if isinstance(x, list):
        return [v for v in x if v not in PLACEHOLDER]
    if pd.isna(x):
        return []
    return [v for v in ast.literal_eval(x) if v not in PLACEHOLDER]


def experience_years(skills):
    """The API mixes 'monthsOfExperience: 36' into the skill list; return the lowest requirement in years."""
    months = [int(s.split(':')[1]) for s in skills if s.startswith('monthsOfExperience')]
    return min(months) / 12 if months else np.nan


def normalize_state(s):
    """Full names -> two-letter codes; anything that is not a US state becomes missing."""
    if pd.isna(s):
        return np.nan
    if s in US_STATES:
        return US_STATES[s]
    return s if len(s) == 2 and s.isupper() else np.nan


def salary_unit(v):
    if config.HOURLY_MIN <= v < config.HOURLY_MAX:
        return 'hourly'
    if v >= config.ANNUAL_MIN:
        return 'annual'
    return 'unknown'


def clean(raw):
    """Raw export -> one row per posting with typed columns. Returns (clean frame, row counts per stage)."""
    counts = {'raw': len(raw)}
    df = raw.drop_duplicates().copy()
    counts['after_exact_duplicates'] = len(df)
    df = df.drop_duplicates(subset=config.DEDUP_KEYS)
    counts['after_repeat_scrapes'] = len(df)
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])

    for c in config.LIST_COLUMNS:
        df[c] = df[c].apply(parse_list)
    df['workPlace'] = df['workPlace'].apply(lambda l: sorted({v.capitalize() for v in l}))
    df['min_exp_years'] = df['skills'].apply(experience_years)
    df['skills'] = df['skills'].apply(lambda sk: [s for s in sk if not s.startswith('monthsOfExperience')])
    df['n_skills'] = df['skills'].str.len()

    unknown = sorted(set(df['state'].dropna()) - set(US_STATES) - {s for s in df['state'].dropna() if len(s) == 2 and s.isupper()})
    if unknown:
        log.info('State values set to missing: %s', unknown)
    df['state'] = df['state'].apply(normalize_state)

    for c in ['dateCreated', 'dateActive']:
        df[c] = pd.to_datetime(df[c], format='ISO8601')
    df['batch'] = df['dateCreated'].dt.strftime('%Y-%m')
    df['active_days'] = (df['dateActive'] - df['dateCreated']).dt.days

    df['seniority'] = df['careerLevel'].apply(lambda l: max((LEVEL[x] for x in l if x in LEVEL), default=np.nan))
    df['seniority_label'] = df['seniority'].map(LABEL)
    df['is_remote'] = df['workPlace'].apply(lambda l: 'Remote' in l)
    df['is_parttime'] = df['workType'].apply(lambda l: 'PartTime' in l)
    df['is_intern'] = df['contractType'].apply(lambda l: 'Internship' in l)

    df['salary_unit'] = df['minSalary'].apply(salary_unit)
    df['salary_annual'] = np.where(df['salary_unit'] == 'hourly', df['minSalary'] * config.HOURS_PER_YEAR,
                                   np.where(df['salary_unit'] == 'annual', df['minSalary'], np.nan))
    df = df.reset_index(drop=True)
    counts['clean'] = len(df)
    return df, counts


def save_clean(df, path):
    """Write the cleaned postings as CSV; list columns are stored as 'a|b|c' (the format app.py reads)."""
    out = df.assign(**{c: df[c].str.join('|') for c in config.LIST_COLUMNS})
    out.to_csv(path, index=False)
    return path
