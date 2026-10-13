"""Machine learning on skills, used by the "Skill ML Insights" tab of the dashboard.
The same methods are explained step by step in skills_ml_experiments.ipynb.
Needs pandas, numpy, scikit-learn, scipy and networkx."""
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold

import skills_analysis as sa

MIN_SKILL_POSTINGS = 8   # a skill needs this many baskets to be modelled


def baskets(df, ambiguous='corroborated'):
    """Posting x skill 0/1 matrix for postings that list at least two curated skills."""
    long = sa.long_skills(df, ambiguous)
    M = long.assign(v=1).pivot_table(index='job_id', columns='skill', values='v', aggfunc='max', fill_value=0)
    B = M.loc[M.sum(axis=1) >= 2]
    B = B.loc[:, B.sum(axis=0) >= MIN_SKILL_POSTINGS]
    return B.loc[B.sum(axis=1) >= 2]


def association_rules(B, min_support=8):
    """Rules 'A -> C' and 'A + B -> C' with support (postings), confidence and lift."""
    sets = [tuple(sorted(B.columns[row.astype(bool)])) for row in B.values]
    counts = {}
    for s in sets:
        for k in (1, 2, 3):
            for combo in itertools.combinations(s, k):
                counts[combo] = counts.get(combo, 0) + 1
    rows = []
    for items, n in counts.items():
        if len(items) < 2 or n < min_support:
            continue
        for cons in items:
            ante = tuple(i for i in items if i != cons)
            conf = n / counts[ante]
            rows.append({'antecedent': ' + '.join(ante), 'consequent': cons, 'size': len(ante), 'postings': n,
                         'confidence': conf, 'lift': conf / (counts[(cons,)] / len(sets))})
    return pd.DataFrame(rows)


def skill_graph(B, min_together=4, min_lift=2.0):
    """Skills as nodes; a link when two skills are posted together often and well above chance."""
    co = B.T @ B
    n = np.diag(co).astype(float)
    lift = co * len(B) / np.outer(n, n)
    G = nx.Graph()
    G.add_nodes_from(B.columns)
    for a, b in itertools.combinations(B.columns, 2):
        if co.loc[a, b] >= min_together and lift.loc[a, b] > min_lift:
            G.add_edge(a, b, weight=float(co.loc[a, b]))
    return G


def communities(B):
    """Louvain communities of the skill graph: (label per skill, modularity, betweenness per skill)."""
    G = skill_graph(B)
    parts = nx.community.louvain_communities(G, weight='weight', seed=0)
    parts = sorted(parts, key=len, reverse=True)
    label = pd.Series({s: i + 1 for i, c in enumerate(parts) for s in c}).reindex(B.columns)
    strength = pd.Series(dict(G.degree(weight='weight'))).reindex(B.columns)
    between = pd.Series(nx.betweenness_centrality(G)).reindex(B.columns)
    return label, nx.community.modularity(G, parts, weight='weight'), between, strength


def lda_topics(B, k=7):
    """LDA topics: (skill weights per topic, topic mix per posting)."""
    lda = LatentDirichletAllocation(n_components=k, random_state=0, max_iter=200, learning_method='batch',
                                    doc_topic_prior=0.2, topic_word_prior=0.1).fit(B.values)
    phi = pd.DataFrame(lda.components_ / lda.components_.sum(axis=1, keepdims=True), columns=B.columns,
                       index=range(1, k + 1))
    theta = pd.DataFrame(lda.transform(B.values), index=B.index, columns=phi.index)
    return phi, theta


def hierarchical_groups(B, cut=0.82):
    """Average-linkage clustering of skills on Jaccard distance, cut into flat groups."""
    L = linkage(pdist(B.T.values.astype(bool), metric='jaccard'), method='average')
    return pd.Series(fcluster(L, t=cut, criterion='distance'), index=B.columns)


def grouping_agreement(B, louvain_label, phi, hier):
    """Adjusted Rand index between four ways of grouping the same skills (1 = identical, 0 = chance)."""
    groups = pd.DataFrame({
        'Manual course areas': pd.Series({s: sa.AREA_OF[s] for s in B.columns}),
        'Network communities (Louvain)': louvain_label,
        'Topic model (LDA)': phi.idxmax(axis=0),
        'Hierarchical clustering': hier,
    })
    names = list(groups.columns)
    ari = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        ari.loc[a, b] = ari.loc[b, a] = adjusted_rand_score(groups[a].astype(str), groups[b].astype(str))
    return ari, groups


def item_similarity(B):
    X = B.values.astype(float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    S = (X.T @ X) / np.outer(norm, norm)
    np.fill_diagonal(S, 0)
    return S


def recommend(have, S, cols, k=5):
    """Next skills given the skills already held (item-based collaborative filtering)."""
    idx = [cols.get_loc(s) for s in have if s in cols]
    if not idx:
        return pd.Series(dtype=float)
    score = S[idx].sum(axis=0)
    score[idx] = -np.inf
    top = np.argsort(score)[::-1][:k]
    return pd.Series(score[top], index=cols[top])


def recommender_hit_rate(B, companies, k=5):
    """Hide one skill per posting; how often is it in the top-k? Returns (recommender, popularity baseline, n)."""
    rng = np.random.default_rng(0)
    hit, hit_pop, total = 0, 0, 0
    splits = min(5, pd.Series(companies).nunique())
    for tr, te in GroupKFold(n_splits=splits).split(B, groups=companies):
        S = item_similarity(B.iloc[tr])
        popular = list(B.iloc[tr].sum().sort_values(ascending=False).index)
        for j in te:
            have = list(B.columns[B.iloc[j].values.astype(bool)])
            hidden = have.pop(rng.integers(len(have)))
            hit += hidden in recommend(have, S, B.columns, k).index
            hit_pop += hidden in [s for s in popular if s not in have][:k]
            total += 1
    return hit / total, hit_pop / total, total
