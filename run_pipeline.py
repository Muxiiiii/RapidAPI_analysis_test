"""Run the whole pipeline: raw API export -> cleaned postings -> scores and skill models.

    python run_pipeline.py                         # uses combined_jobs_data.csv
    python run_pipeline.py --input a.csv b.csv     # stack several raw exports, e.g. a new monthly batch
    python run_pipeline.py --check-only            # clean and validate, write nothing

The Streamlit app (app.py) reads jobs_clean.csv, so a successful run is all the app needs to update."""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring_v2
from pipeline import clean, config, outputs, validate

log = logging.getLogger('pipeline')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', type=Path, default=config.RAW_INPUTS, help='raw CSV export(s)')
    parser.add_argument('--check-only', action='store_true', help='clean and validate only; write no files')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    start = time.time()

    log.info('1/4 Cleaning %s', ', '.join(p.name for p in args.input))
    raw = clean.load_raw(args.input)
    df, counts = clean.clean(raw)
    log.info('    %s raw rows -> %s postings', f"{counts['raw']:,}", f"{counts['clean']:,}")

    log.info('2/4 Checking data quality')
    try:
        checks = validate.validate(df)
    except validate.DataQualityError as e:
        log.error('    %s', e)
        log.error('Stopped. No files were written.')
        return 1
    log.info('    %d checks passed', len(checks))
    if args.check_only:
        log.info('Check only: no files written.')
        return 0

    log.info('3/4 Writing cleaned postings')
    clean.save_clean(df, config.CLEAN_CSV)
    log.info('    %s', config.CLEAN_CSV.name)

    log.info('4/4 Scoring course areas and running skill models')
    files, summary = outputs.write(df, config.OUTPUT_DIR)
    v2 = scoring_v2.write(df, config.OUTPUT_DIR / 'scoring_v2')
    files += [f'scoring_v2/{f}' for f in sorted(p.name for p in (config.OUTPUT_DIR / 'scoring_v2').glob('*.csv')) + ['report_v2.json']]
    summary['robustness'] = {'stable_top10_skills': v2.get('stable_top10_skills'), 'onet_check': v2.get('onet_check')}
    report = {'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'inputs': [p.name for p in args.input], 'row_counts': counts,
              'salary_units': df['salary_unit'].value_counts().to_dict(),
              'posting_dates': {'first': str(df['dateCreated'].min().date()), 'last': str(df['dateCreated'].max().date()),
                                'batches': df['batch'].value_counts().sort_index().to_dict()},
              'checks': checks, **summary, 'outputs': [config.CLEAN_CSV.name] + [f'outputs/{f}' for f in files],
              'seconds': round(time.time() - start, 1)}
    (config.OUTPUT_DIR / 'run_report.json').write_text(json.dumps(report, indent=2, default=str))
    for f in files + ['run_report.json']:
        log.info('    outputs/%s', f)
    log.info('Done in %.1fs. Top course areas: %s', report['seconds'],
             ', '.join(f"{a['area']} ({a['score']:.0f})" for a in summary['top_course_areas'][:3]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
