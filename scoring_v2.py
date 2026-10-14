"""Robustness checks for the opportunity score, following the composite-indicator methods
(OECD & JRC 2008 Handbook on Constructing Composite Indicators):

A. Weight uncertainty: 10,000 weight sets drawn from a Dirichlet distribution; median rank, 5-95% rank
   interval and P(top k) for each skill / course area.
B. Objective weights: entropy, CRITIC (Diakoulaki et al. 1995) and PCA weights, compared with the
   original expert weights and equal weights.
C. Pareto front: non-dominated sorting on demand, salary premium and breadth (no weights at all).
D. External check: does each scheme rank O*NET Hot Technologies higher? (AUC; a concurrent check,
   not a forecast, see the note in the report).
RCA: revealed comparative advantage of each skill / area by industry (Alabdulkareem et al. 2018).

The original outputs are untouched; everything is written to outputs/scoring_v2/.

    python scoring_v2.py              # also run as the last step of run_pipeline.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import skills_analysis as sa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'scoring_v2'
DIMS = ['demand', 'salary', 'trend', 'breadth']
EXPERT = np.array([0.45, 0.30, 0.10, 0.15])
N_DRAWS = 10_000
SEED = 42
RCA_MIN_INDUSTRY = 20
ONET_FILE = ROOT / 'external' / 'onet_29_3_technology_skills.txt'   # O*NET 29.3, public data   # industries with fewer postings are left out of the RCA count

# Our skill name -> O*NET Technology Skills "Example". Only products/languages that O*NET lists by name;
# concepts (DevOps, Machine Learning, Scrum, ...) have no O*NET entry and get no label.
ONET_NAME = {
    'Python': 'Python', 'Microsoft Excel': 'Microsoft Excel', 'SQL': 'Structured query language SQL',
    'Microsoft PowerPoint': 'Microsoft PowerPoint', 'Jira': 'Atlassian JIRA', 'Microsoft Office': 'Microsoft Office software',
    'Kubernetes': 'Kubernetes', 'TypeScript': 'TypeScript', 'Amazon Web Services (AWS)': 'Amazon Web Services AWS software',
    'React': 'React', 'Microsoft Word': 'Microsoft Word', 'MicroStation': 'Bentley MicroStation', 'JavaScript': 'JavaScript',
    'Git': 'Git', 'Microsoft SQL Server': 'Microsoft SQL Server', 'Oracle Database': 'Oracle Database',
    'AutoCAD': 'Autodesk AutoCAD', 'Java': 'Oracle Java', 'GitHub': 'GitHub', 'HTML': 'Hypertext markup language HTML',
    'Linux': 'Linux', 'Windows': 'Microsoft Windows', 'PowerShell': 'Microsoft PowerShell', 'ASP.NET': 'Microsoft ASP.NET',
    'Microsoft Azure': 'Microsoft Azure software', 'Terraform': 'IBM Terraform', 'Angular': 'Google Angular',
    'Figma': 'Figma', 'JSON': 'JavaScript Object Notation JSON', 'GitLab': 'GitLab', 'Jenkins': 'Jenkins CI', 'C++': 'C++',
    '.NET': 'Microsoft .NET Framework', 'Docker': 'Docker', 'CSS': 'Cascading style sheets CSS',
    'Salesforce': 'Salesforce software', 'Unix': 'UNIX', 'TensorFlow': 'TensorFlow', 'Apache Spark': 'Apache Spark',
    'Node.js': 'Node.js', 'macOS': 'Apple macOS', 'Active Directory': 'Microsoft Active Directory',
    'Apache Airflow': 'Apache Airflow', 'Rust': 'Rust programming language', 'Power BI': 'Microsoft Power BI',
    'Tableau': 'Tableau', 'Bash': 'Bash', 'PHP': 'PHP', 'Ansible': 'Ansible software', 'PostgreSQL': 'PostgreSQL',
    'VMware': 'VMware', 'Windows Server': 'Microsoft Windows Server', 'SharePoint': 'Microsoft SharePoint',
    'Vue.js': 'Vue.js', 'MySQL': 'MySQL', 'MongoDB': 'MongoDB', 'ServiceNow': 'ServiceNow',
    'Confluence': 'Atlassian Confluence', 'COBOL': 'Common business oriented language COBOL', 'Google Drive': 'Google Drive',
    'SolidWorks': 'Dassault Systemes SolidWorks', 'Slack': 'Slack', 'Grafana': 'Grafana Labs Grafana Cloud',
    'CRM': 'Customer relationship management CRM software', 'ERP': 'Enterprise resource planning ERP software',
    'VPN': 'Virtual private networking VPN software', 'Microservices': 'Microservices Architecture',
}


def load():
    d = pd.read_csv(ROOT / 'jobs_clean.csv', dtype={'batch': str}, parse_dates=['dateCreated', 'dateActive'])
    for c in ['skills', 'workPlace', 'careerLevel', 'workType', 'contractType']:
        d[c] = d[c].fillna('').str.split('|').apply(lambda x: [v for v in x if v])
    return d


def indicator_matrix(sig):
    """Raw indicators, min-max scaled to 0-1. An unreliable trend (small sample) is set to 0 pp, i.e. neutral."""
    enough = sig['trend_n'] >= sa.TREND_MIN_N
    raw = pd.DataFrame({'demand': sig['jobs'], 'salary': sig['salary_premium_pct'],
                        'trend': sig['trend_pp'].where(enough, 0.0), 'breadth': sig['companies']})
    rng = (raw.max() - raw.min()).replace(0, 1)
    return (raw - raw.min()) / rng


def entropy_weights(X):
    P = (X + 1e-9) / (X + 1e-9).sum()
    e = -(P * np.log(P)).sum() / np.log(len(X))
    d = 1 - e
    return (d / d.sum()).values


def critic_weights(X):
    c = X.std() * (1 - X.corr()).sum()
    return (c / c.sum()).values


def pca_weights(X):
    """OECD Handbook approach: squared loadings of the components with eigenvalue > 1 (at least one),
    each scaled to unit sum, weighted by the variance they explain."""
    Z = (X - X.mean()) / X.std().replace(0, 1)
    val, vec = np.linalg.eigh(np.corrcoef(Z.T.values))
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    keep = max(1, int((val > 1).sum()))
    sq = vec[:, :keep] ** 2
    w = (sq / sq.sum(axis=0)) @ (val[:keep] / val[:keep].sum())
    return w / w.sum(), float(val[0] / val.sum())


def rank_of(score):
    return score.rank(ascending=False, method='min').astype(int)


def dirichlet(R, top_k, alpha=None, n=N_DRAWS, seed=SEED):
    """R: percentile-rank matrix (items x 4). Returns rank statistics over random weight sets."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha if alpha is not None else np.ones(R.shape[1]), size=n)
    S = R.values @ W.T                                  # items x draws
    ranks = (-S).argsort(axis=0).argsort(axis=0) + 1    # 1 = best
    return pd.DataFrame({'median_rank': np.median(ranks, axis=1),
                         'rank_p05': np.percentile(ranks, 5, axis=1),
                         'rank_p95': np.percentile(ranks, 95, axis=1),
                         f'p_top{top_k}': (ranks <= top_k).mean(axis=1)}, index=R.index)


def pareto_layers(X):
    """Non-dominated sorting (higher is better on every column). Layer 1 = Pareto front."""
    V = X.values
    layer = np.zeros(len(V), dtype=int)
    left = set(range(len(V)))
    k = 1
    while left:
        front = [i for i in left if not any(np.all(V[j] >= V[i]) and np.any(V[j] > V[i]) for j in left if j != i)]
        for i in front:
            layer[i] = k
        left -= set(front)
        k += 1
    return pd.Series(layer, index=X.index)


def rca(df, key, ambiguous='corroborated'):
    """RCA_{item, industry} = (item's share in the industry) / (item's share in all postings)."""
    long = sa.long_skills(df, ambiguous).drop_duplicates(['job_id', key])
    ind_n = df['industry'].value_counts()
    big = ind_n[ind_n >= RCA_MIN_INDUSTRY].index
    cnt = long[long['industry'].isin(big)].pivot_table(index=key, columns='industry', values='job_id',
                                                        aggfunc='nunique', fill_value=0)
    share_in = cnt / ind_n.reindex(cnt.columns)
    share_all = long.groupby(key)['job_id'].nunique().reindex(cnt.index) / len(df)
    return share_in.div(share_all, axis=0)


def score_all(sig, top_k):
    R = pd.DataFrame({'demand': sig['jobs'].rank(pct=True), 'salary': sig['salary_premium_pct'].rank(pct=True),
                      'trend': sig['trend_pp'].where(sig['trend_n'] >= sa.TREND_MIN_N).rank(pct=True).fillna(0.5),
                      'breadth': sig['companies'].rank(pct=True)}, index=sig.index)
    X = indicator_matrix(sig)
    w_pca, pc1 = pca_weights(X)
    weights = {'expert': EXPERT, 'equal': np.full(4, 0.25), 'entropy': entropy_weights(X),
               'critic': critic_weights(X), 'pca': w_pca}
    out = sig[['jobs', 'salary_premium_pct', 'trend_pp', 'trend_n', 'companies']].copy()
    for name, w in weights.items():
        out[f'score_{name}'] = (R.values @ w) * 100
        out[f'rank_{name}'] = rank_of(out[f'score_{name}'])
    out = out.join(dirichlet(R, top_k).add_prefix('dir_flat_'))            # any weights at all
    out = out.join(dirichlet(R, top_k, alpha=EXPERT * 20).add_prefix('dir_expert_'))   # near the expert weights
    out['pareto_layer'] = pareto_layers(X[['demand', 'salary', 'breadth']])
    rank_cols = [f'rank_{n}' for n in weights]
    out['rank_spread'] = out[rank_cols].max(axis=1) - out[rank_cols].min(axis=1)
    return out.sort_values('rank_expert'), {k: dict(zip(DIMS, np.round(v, 3))) for k, v in weights.items()}, pc1, X


def onet_check(skill, onet_path):
    o = pd.read_csv(onet_path, sep='\t')
    hot = o.groupby('Example')['Hot Technology'].apply(lambda s: (s == 'Y').any())
    lab = pd.Series({s: hot.get(e) for s, e in ONET_NAME.items() if s in skill.index and e in hot.index}, dtype=float)
    res = {'labelled_skills': int(len(lab)), 'hot': int(lab.sum()), 'not_hot': int((lab == 0).sum()), 'auc': {}}
    for c in [c for c in skill.columns if c.startswith('score_')] + ['jobs', 'salary_premium_pct', 'companies']:
        res['auc'][c] = round(float(roc_auc_score(lab, skill.loc[lab.index, c])), 3)
    return res, lab


def robustness(df, ambiguous='corroborated', onet_path=None):
    """All checks for course areas and skills. Returns {key: (table, rca_matrix)} and a summary report."""
    tables, report = {}, {}
    for key, top_k, min_jobs in [('area', 3, 1), ('skill', 10, 5)]:
        sig = sa.signals(df, key, min_jobs=min_jobs, ambiguous=ambiguous)
        table, weights, pc1, X = score_all(sig, top_k)
        if key == 'skill':
            table.insert(0, 'area', sig['area'])
        R = rca(df, key, ambiguous)
        table['industries_rca_ge1'] = (R >= 1).sum(axis=1).reindex(table.index).fillna(0).astype(int)
        report[key] = {'items': len(table), 'top_k': top_k, 'weights': weights,
                       'pca_first_component_share': round(pc1, 3), 'indicator_correlation': X.corr().round(2).to_dict()}
        if key == 'skill' and onet_path and Path(onet_path).exists():
            res, lab = onet_check(table, onet_path)
            table['onet_hot'] = lab.reindex(table.index)
            report['onet_check'] = res
        k = f'dir_flat_p_top{top_k}'
        report[f'stable_top{top_k}_{key}s'] = table.index[table[k] >= 0.8].tolist()
        tables[key] = (table, R)
    return tables, report


def write(df, out_dir=OUT, onet_path=ONET_FILE):
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, report = robustness(df, onet_path=onet_path)
    for key, (table, R) in tables.items():
        table.round(3).to_csv(out_dir / f'{key}_scores_v2.csv')
        R.round(2).to_csv(out_dir / f'{key}_rca_by_industry.csv')
    (out_dir / 'report_v2.json').write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onet', type=Path, default=ONET_FILE,
                   help='O*NET "Technology Skills.txt" for the external check (method D)')
    a = p.parse_args()
    print(json.dumps(write(load(), OUT, a.onet), indent=2, default=str))


if __name__ == '__main__':
    main()
