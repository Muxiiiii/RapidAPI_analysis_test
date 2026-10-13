"""Tests for the pipeline. Run with:  python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import clean, config, fetch, validate  # noqa: E402


def test_parse_list_drops_placeholders():
    assert clean.parse_list("['Python', 'N/A', 'None']") == ['Python']
    assert clean.parse_list(np.nan) == []


def test_experience_years_takes_lowest_requirement():
    assert clean.experience_years(['Python', 'monthsOfExperience: 36', 'monthsOfExperience: 12']) == 1.0
    assert np.isnan(clean.experience_years(['Python']))


@pytest.mark.parametrize('raw, expected', [('California', 'CA'), ('CA', 'CA'), ('New York City', 'NY')])
def test_normalize_state(raw, expected):
    assert clean.normalize_state(raw) == expected


def test_normalize_state_rejects_non_states():
    assert pd.isna(clean.normalize_state('United States'))
    assert pd.isna(clean.normalize_state(np.nan))


@pytest.mark.parametrize('value, unit', [(15, 'hourly'), (7.25, 'hourly'), (99.9, 'hourly'), (20000, 'annual'),
                                         (85000, 'annual'), (250, 'unknown'), (4000, 'unknown'), (1, 'unknown')])
def test_salary_unit(value, unit):
    assert clean.salary_unit(value) == unit


@pytest.fixture(scope='module')
def cleaned():
    raw = clean.load_raw(config.RAW_INPUTS)
    return clean.clean(raw)


def test_clean_removes_duplicates(cleaned):
    df, counts = cleaned
    assert counts['raw'] >= counts['after_exact_duplicates'] >= counts['after_repeat_scrapes'] == len(df)
    assert not df.duplicated(config.DEDUP_KEYS).any()


def test_clean_output_shape(cleaned):
    df, _ = cleaned
    for col in ['salary_unit', 'salary_annual', 'seniority_label', 'batch', 'min_exp_years', 'is_remote']:
        assert col in df.columns
    assert df['skills'].apply(lambda s: isinstance(s, list)).all()
    assert not df['skills'].apply(lambda s: any(x.startswith('monthsOfExperience') for x in s)).any()


def test_hourly_pay_is_annualized(cleaned):
    df, _ = cleaned
    hourly = df[df['salary_unit'] == 'hourly']
    assert np.allclose(hourly['salary_annual'], hourly['minSalary'] * config.HOURS_PER_YEAR)
    assert df.loc[df['salary_unit'] == 'unknown', 'salary_annual'].isna().all()


def test_real_data_passes_quality_checks(cleaned):
    df, _ = cleaned
    assert all(r['passed'] for r in validate.validate(df))


def test_quality_checks_stop_bad_data(cleaned):
    df, _ = cleaned
    with pytest.raises(validate.DataQualityError):
        validate.validate(df.head(50))   # too few rows


def test_missing_columns_are_reported(tmp_path):
    bad = tmp_path / 'bad.csv'
    pd.DataFrame({'title': ['x']}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match='missing required columns'):
        clean.load_raw([bad])


# ---- API download (no real requests: the API is replaced by a fake) ----
class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload, self.text = status, payload or {}, 'error text'

    def json(self):
        return self._payload


class FakeSession:
    """Serves `pages` of results, then empty pages. `fail_first` makes the first calls return that status."""
    def __init__(self, pages, fail_first=()):
        self.pages, self.fail, self.calls = pages, list(fail_first), []

    def get(self, url, headers, params, timeout):
        self.calls.append({'headers': headers, 'params': dict(params)})
        if self.fail:
            return FakeResponse(self.fail.pop(0))
        i = params['page'] - 1
        result = self.pages[i] if i < len(self.pages) else []
        return FakeResponse(200, {'result': result, 'total': sum(len(p) for p in self.pages)})


def posting(i):
    return {'title': f'Job {i}', 'company': 'Acme', 'skills': ['Python'], 'geoPoint': {'lon': 1.0, 'lat': 2.0}}


def test_fetch_pages_until_results_run_out():
    pages = [[posting(i) for i in range(10)], [posting(i) for i in range(10, 20)], [posting(20)]]
    s = FakeSession(pages)
    jobs = fetch.fetch('key', '2026-04-01', '2026-05-01', pause=0, session=s)
    assert len(jobs) == 21
    assert len(s.calls) == 3                      # stops at the short last page, no extra request


def test_fetch_sends_key_and_standard_filters():
    s = FakeSession([[posting(1)]])
    fetch.fetch('secret', '2026-04-01', '2026-05-01', pause=0, session=s)
    call = s.calls[0]
    assert call['headers']['x-rapidapi-key'] == 'secret'
    assert call['params'] == {'countryCode': 'us', 'format': 'json', 'hasSalary': 'true',
                              'dateCreatedMin': '2026-04-01', 'dateCreatedMax': '2026-05-01', 'page': 1}


def test_fetch_respects_max_pages():
    s = FakeSession([[posting(i) for i in range(10)]] * 50)
    assert len(fetch.fetch('key', 'a', 'b', max_pages=4, pause=0, session=s)) == 40


def test_fetch_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(fetch.time, 'sleep', lambda s: None)
    s = FakeSession([[posting(1)]], fail_first=[429, 500])
    assert len(fetch.fetch('key', 'a', 'b', pause=0, session=s)) == 1


def test_fetch_stops_on_bad_key():
    with pytest.raises(fetch.FetchError, match='rejected the key'):
        fetch.fetch('bad', 'a', 'b', pause=0, session=FakeSession([[posting(1)]], fail_first=[403]))


def test_fetch_needs_key(monkeypatch, tmp_path):
    monkeypatch.delenv('RAPIDAPI_KEY', raising=False)
    assert fetch.main(['--date-min', 'a', '--date-max', 'b', '--out', str(tmp_path / 'x.csv')]) == 1
    assert not (tmp_path / 'x.csv').exists()


def test_downloaded_file_flows_through_cleaning(tmp_path):
    """A flattened API download (extra nested columns, real lists) cleans the same way as the original export."""
    raw = pd.read_csv(config.RAW_INPUTS[0]).head(40)
    raw['jsonLD.description'] = 'long text'       # a column the pipeline does not use
    path = tmp_path / 'download.csv'
    fetch.to_frame(raw.to_dict('records')).to_csv(path, index=False)
    df, _ = clean.clean(clean.load_raw([path]))
    assert 'jsonLD.description' not in df.columns
    assert df['skills'].apply(lambda v: isinstance(v, list)).all()
