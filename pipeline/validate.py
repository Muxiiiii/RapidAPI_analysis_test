"""Step 2: data quality checks. A failed check stops the run before anything downstream is written."""
import pandas as pd

import skills_analysis as sa

from . import config


class DataQualityError(Exception):
    pass


def check(df):
    """Return a list of {name, value, limit, passed}."""
    with_skill = sa.long_skills(df)['job_id'].nunique() / len(df) if len(df) else 0
    dates = pd.to_datetime(df['dateCreated'])
    results = [
        ('rows', len(df), f'>= {config.MIN_ROWS}', len(df) >= config.MIN_ROWS),
        ('unknown_salary_share', round((df['salary_unit'] == 'unknown').mean(), 4), f'<= {config.MAX_UNKNOWN_SALARY}',
         (df['salary_unit'] == 'unknown').mean() <= config.MAX_UNKNOWN_SALARY),
        ('postings_with_curated_skill_share', round(with_skill, 4), f'>= {config.MIN_WITH_SKILLS}',
         with_skill >= config.MIN_WITH_SKILLS),
        ('duplicate_postings', int(df.duplicated(config.DEDUP_KEYS).sum()), '== 0',
         not df.duplicated(config.DEDUP_KEYS).any()),
        ('missing_title_or_company', int(df[['title', 'company']].isna().any(axis=1).sum()), '== 0',
         not df[['title', 'company']].isna().any(axis=1).any()),
        ('state_codes_valid', int((df['state'].dropna().str.len() != 2).sum()), '== 0',
         df['state'].dropna().str.len().eq(2).all()),
        ('annual_salary_range', f"{df['salary_annual'].min():,.0f} to {df['salary_annual'].max():,.0f}",
         '10,000 to 1,000,000', df['salary_annual'].dropna().between(10_000, 1_000_000).all()),
        ('dates_parsed', int(dates.isna().sum()), '== 0', not dates.isna().any()),
    ]
    return [{'name': n, 'value': v, 'limit': l, 'passed': bool(p)} for n, v, l, p in results]


def validate(df):
    results = check(df)
    failed = [r for r in results if not r['passed']]
    if failed:
        raise DataQualityError('Data quality checks failed: ' + '; '.join(
            f"{r['name']} = {r['value']} (expected {r['limit']})" for r in failed))
    return results
