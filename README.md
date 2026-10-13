# Arusto Job Market Intelligence

Job market analytics for the question **"what courses should we build next?"**, using job postings collected from the
Daily International Job Postings API (RapidAPI). The repository holds a reproducible data pipeline, a Streamlit
dashboard, and the analysis notebooks behind them.

## What is here

| Path | What it is |
|---|---|
| `run_pipeline.py` | One command that runs the whole pipeline |
| `pipeline/` | The pipeline steps: `fetch.py` (optional API download), `clean.py`, `validate.py`, `outputs.py`, and settings in `config.py` |
| `skills_analysis.py` | Skill taxonomy (13 course areas), market signals and the opportunity score |
| `skills_ml.py` | Skill models: association rules, communities, topics, recommender and others |
| `app.py` | The Streamlit dashboard |
| `combined_jobs_data.csv` | Raw API export (pipeline input) |
| `jobs_clean.csv` | Cleaned postings (pipeline output, read by the dashboard) |
| `outputs/` | Score tables, model tables and `run_report.json` from the last run |
| `tests/` | Tests for the cleaning rules and the data quality checks |
| `*.ipynb` | Analysis notebooks: the full walk-through and the model experiments |

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

The pipeline runs four steps and stops at the first problem:

1. **Clean** the raw export: remove duplicates and repeat scrapes, parse list columns, standardize states and
   seniority, infer whether each salary is hourly or annual and annualize hourly pay.
2. **Check data quality**: enough rows, few salaries with an unknown unit, enough postings with skills, no duplicates,
   valid states and dates. If a check fails, nothing is written.
3. **Write** `jobs_clean.csv`.
4. **Score and model**: opportunity scores for course areas and skills, association rules, skill communities and the
   recommender's tested hit rate, plus a `run_report.json` that records what was run and what came out.

## Download new postings from the API (optional)

Needs a RapidAPI key for the Daily International Job Postings API. The key is read from the environment and is never
written to a file.

```bash
export RAPIDAPI_KEY=your-key
python -m pipeline.fetch --date-min 2026-04-01 --date-max 2026-05-01 --out raw_2026-04.csv
```

Each page of results is one API request and counts against the plan's quota; `--max-pages` (default 30) caps a run.
Every batch uses the same filters (US postings that state a salary) so that batches can be compared. Without a key,
skip this step: the pipeline runs on any CSV export.

## Add new data

Put the new raw export next to the existing one and pass both files. They are stacked and de-duplicated together.

```bash
python run_pipeline.py --input combined_jobs_data.csv new_batch.csv
```

To check a file without changing anything:

```bash
python run_pipeline.py --input new_batch.csv --check-only
```

Commit the updated `jobs_clean.csv` and `outputs/` and push. The Streamlit app redeploys from the repository on its own.

## Tests

```bash
pip install pytest
python -m pytest
```

GitHub Actions runs the tests and the data quality checks on every push (`.github/workflows/pipeline.yml`).

## Settings

Thresholds live in `pipeline/config.py`: the salary cut-offs used to tell hourly from annual pay, the columns that are
dropped, and the data quality limits. The skill taxonomy lives in `skills_analysis.py`.

## Known limits

- The API gives no salary unit, so hourly and annual pay are inferred from the size of the number.
- Postings come from several job boards in different proportions per batch. Most tech postings come from one board
  (devjobs), which the dashboard lets you exclude.
- The data covers four collection batches from September 2025 to March 2026, which is too few and too uneven to
  measure a trend.
- The API download step is covered by tests against a simulated API, but has not been run against the live API.
- The existing batches were collected with different filters and page ranges, so they differ in make-up.
