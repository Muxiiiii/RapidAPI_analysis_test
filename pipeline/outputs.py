"""Step 3: course-area and skill scores. Step 4: skill models. Both write tables to the outputs folder."""
import pandas as pd

import skills_analysis as sa
import skills_ml as ml

DEVJOBS = 'devjobs_us'


def score_tables(df):
    """Opportunity scores for course areas and skills, with the context needed to judge them."""
    area = sa.opportunity(sa.signals(df, 'area'))
    mix = sa.source_mix(df)
    area['devjobs_pct'] = (mix.get(DEVJOBS, 0) / mix.sum(axis=1) * 100).reindex(area.index).fillna(0)
    area['rank'] = range(1, len(area) + 1)
    without = sa.opportunity(sa.signals(df[df['source'] != DEVJOBS], 'area'))
    area['rank_without_devjobs'] = pd.Series(range(1, len(without) + 1), index=without.index).reindex(area.index)
    level = (area['jobs'] >= 100).astype(int) + (area['jobs'] >= 40).astype(int) - (area['devjobs_pct'] >= 70).astype(int)
    area['confidence'] = level.clip(lower=0).map({0: 'Low', 1: 'Medium', 2: 'High'})
    skill = sa.opportunity(sa.signals(df, 'skill', min_jobs=5))
    return area, skill


def model_tables(df):
    """Association rules, skill communities and the recommender's tested hit rate."""
    B = ml.baskets(df)
    rules = ml.association_rules(B).sort_values(['confidence', 'postings'], ascending=False)
    label, modularity, between, strength = ml.communities(B)
    comm = pd.DataFrame({'community': label, 'bridge_score': between, 'link_strength': strength,
                         'course_area': pd.Series({s: sa.AREA_OF[s] for s in B.columns})}).rename_axis('skill')
    hit, hit_popular, n = ml.recommender_hit_rate(B, df.loc[B.index, 'company'].values)
    metrics = {'postings_with_2plus_skills': int(len(B)), 'skills_modelled': int(B.shape[1]),
               'rules': int(len(rules)), 'communities': int((label.value_counts() >= 2).sum()),
               'modularity': round(float(modularity), 3), 'recommender_top5_hit_rate': round(hit, 3),
               'popularity_baseline_hit_rate': round(hit_popular, 3), 'recommender_test_postings': int(n)}
    return rules, comm.sort_values(['community', 'link_strength'], ascending=[True, False]), metrics


def write(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    area, skill = score_tables(df)
    rules, comm, metrics = model_tables(df)
    files = {'course_area_scores.csv': area.round(3), 'skill_scores.csv': skill.round(3),
             'association_rules.csv': rules.round(3), 'skill_communities.csv': comm.round(4)}
    for name, table in files.items():
        table.to_csv(out_dir / name, index=name != 'association_rules.csv')
    summary = {'top_course_areas': [{'area': a, 'score': round(float(r['score']), 1), 'postings': int(r['jobs']),
                                     'confidence': r['confidence']} for a, r in area.head(5).iterrows()],
               'skill_models': metrics}
    return list(files), summary
