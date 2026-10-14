"""Arusto job market dashboard. Run: streamlit run app.py
Data is jobs_clean.csv, the cleaned output of jobs_analysis.ipynb."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import scoring_v2 as rb
import skills_analysis as sa
import skills_ml as ml

BLUE, GRAY, INK = '#2a78d6', '#c3c2b7', '#52514e'
# Fixed colour per job board, so a board keeps its colour whatever is filtered
SOURCE_COLORS = {'devjobs': '#8ec5f4', 'linkedin': '#c9b3f1', 'careerjet': '#9ddcc1', 'greenhouse': '#f3d88f',
                 'Other boards': '#cfcec8'}
LIST_COLS = ['skills', 'workPlace', 'careerLevel', 'workType', 'contractType']
DEVJOBS = 'devjobs_us'
st.set_page_config(page_title='Arusto Job Market Intelligence', layout='wide')


@st.cache_data
def load():
    d = pd.read_csv('jobs_clean.csv', dtype={'batch': str})
    for c in LIST_COLS:   # list columns are stored as 'a|b|c'
        d[c] = d[c].fillna('').apply(lambda s: s.split('|') if s else [])
    return d


@st.cache_data
def load_signals(mode, without_devjobs):
    d = load()
    if without_devjobs:
        d = d[d['source'] != DEVJOBS]
    return sa.signals(d, 'area', ambiguous=mode), sa.signals(d, 'skill', min_jobs=5, ambiguous=mode)


@st.cache_data
def load_mix(mode):
    return sa.source_mix(load(), mode)


@st.cache_data
def load_partners(skill, mode, without_devjobs):
    d = load()
    if without_devjobs:
        d = d[d['source'] != DEVJOBS]
    return sa.partners(d, skill, mode)


@st.cache_data
def load_ml(mode, without_devjobs):
    """Everything the Skill ML Insights tab needs, computed once per setting."""
    d = load()
    if without_devjobs:
        d = d[d['source'] != DEVJOBS]
    B = ml.baskets(d, mode)
    label, modularity, between, strength = ml.communities(B)
    phi, theta = ml.lda_topics(B)
    ari, groups = ml.grouping_agreement(B, label, phi, ml.hierarchical_groups(B))
    hit, hit_pop, n_test = ml.recommender_hit_rate(B, d.loc[B.index, 'company'].values)
    title_scores, title_sugg = ml.title_classifier(d, mode)
    return dict(B=B, rules=ml.association_rules(B), label=label, modularity=modularity, between=between,
                strength=strength, phi=phi, theta=theta, ari=ari, groups=groups, hit=hit, hit_pop=hit_pop,
                n_test=n_test, S=ml.item_similarity(B), salary=d.loc[B.index, 'salary_annual'],
                sim=ml.embedding_similarity(B), title_scores=title_scores, title_sugg=title_sugg)


@st.cache_data
def load_robustness(mode, without_devjobs):
    """Weight-sampling, objective-weight, Pareto, RCA and O*NET checks for the opportunity score."""
    d = load()
    if without_devjobs:
        d = d[d['source'] != DEVJOBS]
    return rb.robustness(d, mode, rb.ONET_FILE)


def confidence(postings, devjobs_pct):
    """Plain-language reliability of an area's numbers: sample size, then dependence on one job board."""
    level = 2 if postings >= 100 else 1 if postings >= 40 else 0
    if devjobs_pct >= 70:
        level -= 1
    return ['Low', 'Medium', 'High'][max(level, 0)]


def money(v):
    return '' if pd.isna(v) else f'${v:,.0f}'


def layout(fig, title, subtitle=None, height=None):
    text = f'{title}<br><span style="font-size:12px;color:{INK}">{subtitle}</span>' if subtitle else title
    fig.update_layout(title=dict(text=text, font_size=15), margin=dict(l=0, r=10, t=64 if subtitle else 44, b=0),
                      height=height, yaxis_title=None)
    return fig


df = load()
dates = pd.to_datetime(df['dateCreated'], format='ISO8601')
MODES = {'Count only when a specific skill confirms them (recommended)': 'corroborated',
         'Count all': 'keep', 'Exclude all': 'drop'}

# ---- Global settings (apply to Course Opportunities and Skill Bundles) ----
st.sidebar.header('Global settings')
st.sidebar.caption('Apply to every tab except Salary Overview, which has its own filters.')
without_devjobs = st.sidebar.toggle('Exclude devjobs postings', value=False,
                                    help='devjobs is a tech-only job board that supplies most tech postings in this '
                                         'data. Turn this on to see whether a result depends on that one board.')
mode = MODES[st.sidebar.radio('Broad skill tags (cloud, security, ai, network, ...)', list(MODES),
                              help='These tags are too broad to trust alone. By default one is counted only when the '
                                   'same posting also lists a specific skill from the same course area.')]
st.sidebar.divider()
st.sidebar.markdown(f'**About the data**  \n{len(df):,} US job postings after de-duplication  \n'
                    f'Posted {dates.min():%b %Y} to {dates.max():%b %Y}, collected in 4 batches  \n'
                    f'Most recent posting: {dates.max():%b %d, %Y}')

st.title('Arusto Job Market Intelligence')
st.caption('Which courses should we build next? Evidence from US job postings on what employers ask for. '
           f'Postings from {dates.min():%b %Y} to {dates.max():%b %Y}.')

AREA_COLS = {'score': 'Score', 'jobs': 'Postings', 'companies': 'Companies', 'median_salary': 'Median salary',
             'salary_premium_pct': 'Salary difference', 'change': 'Two-period change', 'devjobs_pct': 'From devjobs',
             'confidence': 'Confidence', 'rank_wo': 'Rank without devjobs'}


def area_table(sig_all, sig_wo, mix, weights):
    """Scores plus the context needed to judge them: source dependence, confidence, rank without devjobs."""
    area = sa.opportunity(sig_all, *weights)
    wo = sa.opportunity(sig_wo, *weights)
    area['devjobs_pct'] = (mix.get(DEVJOBS, 0) / mix.sum(axis=1) * 100).reindex(area.index).fillna(0)
    area['confidence'] = [confidence(j, p) for j, p in zip(area['jobs'], area['devjobs_pct'])]
    area['rank'] = range(1, len(area) + 1)
    area['rank_wo'] = pd.Series(range(1, len(wo) + 1), index=wo.index).reindex(area.index)
    return area


def course_tab():
    sig_all, skill_all = load_signals(mode, False)
    sig_wo, skill_wo = load_signals(mode, True)
    mix = load_mix(mode)

    presets = {'Original': dict(zip(rb.DIMS, rb.EXPERT)), 'Equal': dict.fromkeys(rb.DIMS, 0.25)}
    presets.update({n.upper() if n == 'critic' else n.title(): w
                    for n, w in load_robustness(mode, without_devjobs)[1]['area']['weights'].items()
                    if n in ('critic', 'entropy')})
    keys = {'demand': 'w_demand', 'salary': 'w_salary', 'breadth': 'w_breadth', 'trend': 'w_trend'}

    def apply_preset():
        for dim, k in keys.items():   # sliders move in steps of 0.05
            st.session_state[k] = round(float(presets[st.session_state['w_preset']][dim]) * 20) / 20

    for dim, k in keys.items():
        st.session_state.setdefault(k, round(float(presets['Original'][dim]) * 20) / 20)
    with st.expander('Adjust scoring weights', expanded=True):
        st.caption('The opportunity score ranks each area on four signals and averages the ranks with these weights. '
                   'Pick a preset or move the sliders; the Score Robustness tab explains the presets.')
        st.radio('Preset', list(presets), key='w_preset', horizontal=True, on_change=apply_preset,
                 help='Original: weights we set ourselves. Equal: 25% each. CRITIC and Entropy: weights set from '
                      'the data (CRITIC discounts signals that repeat each other).')
        w = st.columns(4)
        wd = w[0].slider('Hiring demand', 0.0, 1.0, step=0.05, key='w_demand',
                         help='Number of postings asking for skills in the area')
        ws = w[1].slider('Salary difference', 0.0, 1.0, step=0.05, key='w_salary',
                         help='Pay of postings in the area compared with other postings of the same seniority and job board')
        wb = w[2].slider('Employer coverage', 0.0, 1.0, step=0.05, key='w_breadth', help='Number of distinct companies hiring')
        wt = w[3].slider('Two-period change', 0.0, 1.0, step=0.05, key='w_trend',
                         help='Change in share of postings between Sep 2025 and Mar 2026. Neutral where the sample is too small.')
    top = st.container()
    weights = (wd, ws, wt, wb)
    full = area_table(sig_all, sig_wo, mix, weights)
    # the view follows the global devjobs switch; `full` always keeps both rankings for comparison
    area = area_table(sig_wo, sig_wo, mix, weights) if without_devjobs else full
    if without_devjobs:
        area['devjobs_pct'] = full['devjobs_pct'].reindex(area.index)
        area['confidence'] = [confidence(j, 0) for j in area['jobs']]
    st.session_state['area_scores'], st.session_state['weights'] = area, weights

    with top:
        st.subheader('Three course areas worth validating next')
        st.caption('Ranked by opportunity score' + (', with devjobs postings excluded.' if without_devjobs else '.')
                   + ' These are leads to test with employers and learners, not final decisions.')
        for col, (name, r) in zip(st.columns(3), area.head(3).iterrows()):
            with col.container(border=True):
                st.markdown(f'**{int(r["rank"])}. {name}**')
                st.markdown(f'Score **{r["score"]:.0f}** / 100 · Confidence: **{r["confidence"]}**')
                diff = r['salary_premium_pct']
                lines = [f'{int(r["jobs"])} postings from {int(r["companies"])} companies',
                         f'Pays {abs(diff):.0f}% {"more" if diff >= 0 else "less"} than comparable postings']
                if without_devjobs:
                    lines.append(f'Rank with all job boards: {int(full.loc[name, "rank"])}')
                else:
                    lines.append(f'{r["devjobs_pct"]:.0f}% of postings come from devjobs')
                    wo_rank = r['rank_wo']
                    lines.append('Rank without devjobs: ' + ('not enough data' if pd.isna(wo_rank) else str(int(wo_rank))))
                st.markdown('\n'.join(f'- {x}' for x in lines))

        top3 = set(area.index[:3])

        left, right = st.columns(2)
        with left:
            g = area.sort_values('score')
            fig = go.Figure(go.Bar(x=g['score'], y=g.index, orientation='h', text=g['score'].round().astype(int),
                                   textposition='outside', cliponaxis=False,
                                   marker_color=[BLUE if a in top3 else GRAY for a in g.index],
                                   customdata=g[['jobs', 'companies', 'confidence']],
                                   hovertemplate='<b>%{y}</b><br>Score %{x:.0f}<br>%{customdata[0]} postings, '
                                                 '%{customdata[1]} companies<br>Confidence: %{customdata[2]}<extra></extra>'))
            fig.update_xaxes(range=[0, 108], title='Opportunity score (0 to 100)')
            st.plotly_chart(layout(fig, 'Opportunity score', height=30 * len(g) + 90), use_container_width=True)
        with right:
            parts = {'rank_demand': 'Demand', 'rank_salary': 'Salary', 'rank_breadth': 'Employers',
                     'rank_trend': 'Change'}
            h = area.sort_values('score')[list(parts)].rename(columns=parts).round()
            fig = go.Figure(go.Heatmap(z=h.values, x=h.columns, y=h.index, zmin=0, zmax=100, xgap=2, ygap=2,
                                       colorscale=[[0, '#cde2fb'], [1, '#184f95']], showscale=False,
                                       text=h.values.astype(int), texttemplate='%{text}',
                                       hovertemplate='<b>%{y}</b><br>%{x}: %{z:.0f} / 100<extra></extra>'))
            fig.update_xaxes(side='bottom', tickangle=0)
            st.plotly_chart(layout(fig, 'What drives each score',
                                   'Rank on each signal: 0 = weakest area, 100 = strongest.',
                                   height=30 * len(h) + 110), use_container_width=True)

        # Demand vs salary difference: label and highlight only the top three
        g = area.reset_index()
        g['label'] = [a if i < 3 else '' for i, a in enumerate(g['area'])]
        g['pos'] = ['bottom center' if i == 1 else 'top center' for i in range(len(g))]
        fig = go.Figure(go.Scatter(
            x=g['jobs'], y=g['salary_premium_pct'], mode='markers+text', text=g['label'], textposition=g['pos'],
            marker=dict(size=g['companies'], sizemode='area', sizeref=2 * g['companies'].max() / 44 ** 2, sizemin=5,
                        color=[BLUE if a in top3 else GRAY for a in g['area']], line=dict(color='white', width=2)),
            customdata=g[['area', 'companies', 'median_salary', 'confidence']],
            hovertemplate='<b>%{customdata[0]}</b><br>%{x} postings, %{customdata[1]} companies<br>'
                          'Salary difference %{y:+.0f}%<br>Median salary $%{customdata[2]:,.0f}<br>'
                          'Confidence: %{customdata[3]}<extra></extra>'))
        fig.add_hline(y=0, line_width=1, line_color='#898781')
        fig.update_xaxes(title='Postings asking for skills in the area', range=[-g['jobs'].max() * 0.08, g['jobs'].max() * 1.22])
        fig.update_yaxes(title='Salary difference vs comparable postings', ticksuffix='%')
        layout(fig, 'Hiring demand and salary difference', 'Bubble size = number of companies hiring. Hover for details.',
               height=430)
        fig.update_layout(yaxis_title='Salary difference vs comparable postings')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('**How to read salary difference.** It compares the posted minimum salary of postings that ask for '
                   'skills in the area with other postings at the same seniority level and on the same job board. '
                   'Industry and location are not controlled. It describes which jobs pay more; it is not the raise '
                   'someone gets from learning the skill.')

        # Where the postings come from
        m = mix.reindex(area.sort_values('score').index).fillna(0)
        m.columns = [c.replace('_us', '') for c in m.columns]
        main = [c for c in SOURCE_COLORS if c in m.columns]
        m['Other boards'] = m.drop(columns=main).sum(axis=1)
        share = m[main + ['Other boards']].div(m[main + ['Other boards']].sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for s in main + ['Other boards']:
            fig.add_bar(x=share[s], y=[f'{a}  (n={int(n)})' for a, n in zip(share.index, m[main + ['Other boards']].sum(axis=1))],
                        orientation='h', name=s, marker_color=SOURCE_COLORS[s], marker_line=dict(color='white', width=2),
                        customdata=m[s], hovertemplate=f'<b>{s}</b><br>%{{x:.0f}}% (%{{customdata:.0f}} postings)<extra></extra>')
        fig.update_layout(barmode='stack', legend=dict(orientation='h', y=-0.12, traceorder='normal'))
        fig.update_xaxes(ticksuffix='%', range=[0, 100], title=None)
        st.plotly_chart(layout(fig, 'Where each area\'s postings come from',
                               'Share of postings by job board, all boards included. An area dominated by one board '
                               'reflects that board as much as the market.', height=30 * len(share) + 150),
                        use_container_width=True)

        st.subheader('All course areas')
        t = area.copy()
        t['change'] = [f'{v:+.1f} pp' if ok else 'not enough data' for v, ok in zip(t['trend_pp'], t['trend_reliable'])]
        show = pd.DataFrame({
            'Score': t['score'].round().astype(int), 'Confidence': t['confidence'], 'Postings': t['jobs'].astype(int),
            'Companies': t['companies'].astype(int), 'Median salary': t['median_salary'].map(money),
            'Salary difference': t['salary_premium_pct'].map('{:+.0f}%'.format), 'Two-period change': t['change'],
            'From devjobs': t['devjobs_pct'].map('{:.0f}%'.format)})
        if not without_devjobs:
            show['Rank without devjobs'] = t['rank_wo'].map(lambda v: '' if pd.isna(v) else str(int(v)))
        st.dataframe(show.rename_axis('Course area'), use_container_width=True)
        st.caption('**Confidence** is High with 100 or more postings, Medium with 40 to 99, Low below 40, and drops '
                   'one level when 70% or more of the postings come from devjobs. **Two-period change** compares the '
                   'share of postings in Sep 2025 with Mar 2026 on LinkedIn and Greenhouse only. It is a two-point '
                   'comparison, not a continuous growth trend, and is shown only where at least 20 postings are involved.')

        st.caption('For the skills, pay and employers behind one area, open the **Area Deep Dive** tab.')

        with st.expander('What this dashboard cannot tell you'):
            st.markdown(
                'Job postings show what employers ask for. They do not show who would buy a course. Before committing '
                'to an area, add what this data lacks:\n'
                '- **Target learners**: who would take the course and whether they can pay for it\n'
                '- **Entry barrier**: how much a learner must already know\n'
                '- **Existing coverage**: whether Arusto or competitors already teach it\n'
                '- **Production cost**: how expensive the course is to build and keep current\n\n'
                'An area with a negative salary difference, such as Office Productivity, can still be a good course '
                'for beginners: its value is breadth of demand, not high-paying jobs.')
        with st.expander('How the skill tags were cleaned'):
            st.markdown(
                '- **Real skills only.** 139 reviewed tags are kept. Generic words, mislabels and job titles '
                '(product manager, project manager) are removed.\n'
                '- **Standard names.** Tags are shown with standardized names; the raw tag is kept in the data.\n'
                '- **No merging of distinct skills.** Only spelling variants are merged. SQL and MySQL, Git and GitHub, '
                'Angular and AngularJS, AI and Machine Learning stay separate.\n'
                '- **Broad tags are checked.** The table shows how many postings with each broad tag also list a specific '
                'skill from the same course area.')
            amb = sa.ambiguous_check(df)
            st.dataframe(pd.DataFrame({'Postings with tag': amb['postings_with_tag'], 'Confirmed': amb['corroborated'],
                                       'Confirmed share': amb['corroborated_pct'].map('{:.0f}%'.format)})
                         .rename_axis('Broad tag'), use_container_width=True)


@st.cache_data
def load_area_postings(mode, without_devjobs):
    """One row per (posting, course area), with the posting's own columns."""
    d = load()
    if without_devjobs:
        d = d[d['source'] != DEVJOBS]
    long = sa.long_skills(d, mode)
    return d, long


def area_tab():
    area = st.session_state.get('area_scores')
    weights = st.session_state.get('weights')
    if area is None:
        st.info('Open the Course Opportunities tab first.')
        return
    d, long = load_area_postings(mode, without_devjobs)
    st.markdown('Pick one course area to see what is behind its score: which skills employers ask for, what the jobs '
                'pay, who is hiring and at what level.')
    focus = st.selectbox('Course area', area.index.tolist(), key='focus_area')
    r = area.loc[focus]
    ids = long.loc[long['area'] == focus, 'job_id'].unique()
    jobs = d.loc[ids]
    paid = jobs.dropna(subset=['salary_annual'])

    # ---- Headline ----
    k = st.columns(5)
    k[0].metric('Rank', f"{int(r['rank'])} of {len(area)}", help='By opportunity score, with the current weights')
    k[1].metric('Opportunity score', f"{r['score']:.0f} / 100")
    k[2].metric('Postings', f"{int(r['jobs']):,}", help=f"{int(r['companies'])} different companies")
    k[3].metric('Median salary', money(r['median_salary']), delta=f"{r['salary_premium_pct']:+.0f}% vs similar jobs",
                help='Annualized posted minimum. The comparison is with other postings at the same seniority and job board.')
    k[4].metric('Confidence', r['confidence'], help=f"{r['devjobs_pct']:.0f}% of these postings come from devjobs")

    # ---- Why this score ----
    left, right = st.columns(2)
    with left:
        parts = {'rank_demand': 'Hiring demand', 'rank_salary': 'Salary difference',
                 'rank_breadth': 'Employer coverage', 'rank_trend': 'Two-period change'}
        vals = r[list(parts)].astype(float)
        avg = area[list(parts)].mean()
        order = list(parts)[::-1]
        fig = go.Figure()
        fig.add_bar(x=vals[order], y=[parts[p] for p in order], orientation='h', marker_color=BLUE, name=focus,
                    text=vals[order].round().astype(int), textposition='outside', cliponaxis=False,
                    hovertemplate='<b>%{y}</b><br>Rank %{x:.0f} of 100<extra></extra>')
        fig.add_scatter(x=avg[order], y=[parts[p] for p in order], mode='markers', name='Average of all areas',
                        marker=dict(symbol='line-ns', size=22, line=dict(width=2.5, color='#898781')), hoverinfo='skip')
        fig.update_xaxes(range=[0, 112], title='Rank among all course areas (100 = strongest)')
        fig.update_layout(legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(layout(fig, 'Why it scores this way', 'How the area ranks on each of the four signals.', height=330),
                        use_container_width=True)
        best, worst = vals.idxmax(), vals.idxmin()
        st.caption(f'Strongest on **{parts[best].lower()}** (rank {vals[best]:.0f}), weakest on '
                   f'**{parts[worst].lower()}** (rank {vals[worst]:.0f}).'
                   + ('' if r['trend_reliable'] else ' Two-period change has too few postings here, so it is set to a neutral 50.'))
    with right:
        sk = long[long['area'] == focus].groupby('skill').agg(postings=('job_id', 'nunique'), companies=('company', 'nunique'),
                                                              salary=('salary_annual', 'median'))
        g = sk.sort_values('postings').tail(12)
        fig = go.Figure(go.Bar(x=g['postings'], y=g.index, orientation='h', marker_color=BLUE, text=g['postings'],
                               textposition='outside', cliponaxis=False, customdata=g[['companies', 'salary']],
                               hovertemplate='<b>%{y}</b><br>%{x} postings, %{customdata[0]} companies<br>'
                                             'Median salary $%{customdata[1]:,.0f}<extra></extra>'))
        fig.update_xaxes(title='Postings')
        st.plotly_chart(layout(fig, 'Skills employers ask for most', 'What a course in this area would need to cover.',
                               height=max(330, 26 * len(g) + 110)), use_container_width=True)

    # ---- Pay ----
    st.subheader('What the jobs pay')
    left, right = st.columns(2)
    with left:
        allpaid = d.dropna(subset=['salary_annual'])
        rows = []
        for label, part in [(focus, paid), ('All postings', allpaid)]:
            for unit, name in [('annual', 'Salaried'), ('hourly', 'Hourly')]:
                s = part.loc[part['salary_unit'] == unit, 'salary_annual']
                if len(s) >= 5:
                    rows.append({'group': f'{name}: {label}  (n={len(s)})', 'is_focus': label == focus,
                                 'q25': s.quantile(.25), 'median': s.median(), 'q75': s.quantile(.75)})
        q = pd.DataFrame(rows)[::-1]
        fig = go.Figure(go.Scatter(
            x=q['median'], y=q['group'], mode='markers',
            marker=dict(size=12, color=[BLUE if f else '#898781' for f in q['is_focus']]),
            error_x=dict(type='data', symmetric=False, array=q['q75'] - q['median'], arrayminus=q['median'] - q['q25'],
                         color='#898781', thickness=2, width=0),
            customdata=q[['q25', 'q75']],
            hovertemplate='<b>%{y}</b><br>Median $%{x:,.0f}<br>Middle half: $%{customdata[0]:,.0f} to $%{customdata[1]:,.0f}<extra></extra>'))
        fig.update_xaxes(tickprefix='$', tickformat=',.0f', rangemode='tozero', title='Annualized posted minimum salary')
        st.plotly_chart(layout(fig, 'Pay compared with all postings',
                               'Dot = median. Line = middle half of postings. Hourly pay is annualized at 2,080 hours.',
                               height=300), use_container_width=True)
    with right:
        order = ['Intern', 'Junior', 'Medior', 'Senior', 'Lead', 'Director', 'Chief']
        lv = paid.groupby('seniority_label')['salary_annual'].agg(n='size', median='median').reindex(order).dropna()
        lv = lv[lv['n'] >= 3]
        if lv.empty:
            st.info('Too few postings state a seniority level to show pay by level.')
        else:
            fig = go.Figure(go.Bar(x=lv.index, y=lv['median'], marker_color=BLUE, customdata=lv['n'],
                                   text=[f'${v / 1000:.0f}k' for v in lv['median']], textposition='outside', cliponaxis=False,
                                   hovertemplate='<b>%{x}</b><br>Median $%{y:,.0f}<br>%{customdata} postings<extra></extra>'))
            fig.update_yaxes(tickprefix='$', tickformat=',.0f', title=None, range=[0, lv['median'].max() * 1.18])
            st.plotly_chart(layout(fig, 'Median pay by seniority level', 'Levels with at least 3 postings.', height=300),
                            use_container_width=True)

    # ---- Who is hiring ----
    st.subheader('Who is hiring, and at what level')
    c1, c2, c3 = st.columns(3)

    def share_bar(series, title, subtitle, top=8):
        s = series.value_counts().head(top)[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation='h', marker_color=BLUE, text=s.values,
                               textposition='outside', cliponaxis=False,
                               hovertemplate='<b>%{y}</b><br>%{x} postings<extra></extra>'))
        fig.update_xaxes(title='Postings', range=[0, s.max() * 1.2])
        return layout(fig, title, subtitle, height=max(300, 28 * len(s) + 110))

    with c1:
        st.plotly_chart(share_bar(jobs['company'], 'Top hiring companies', f"{jobs['company'].nunique()} companies in total."),
                        use_container_width=True)
    with c2:
        st.plotly_chart(share_bar(jobs['industry'], 'Industries', 'Where the demand comes from.'), use_container_width=True)
    with c3:
        lv = jobs['seniority_label'].fillna('Not stated')
        cmp = pd.DataFrame({'area': lv.value_counts(normalize=True) * 100,
                            'all': d['seniority_label'].fillna('Not stated').value_counts(normalize=True) * 100})
        cmp = cmp.reindex(['Not stated', 'Chief', 'Director', 'Lead', 'Senior', 'Medior', 'Junior', 'Intern']).dropna(how='all').fillna(0)
        fig = go.Figure()
        fig.add_bar(x=cmp['area'], y=cmp.index, orientation='h', marker_color=BLUE, name=focus,
                    hovertemplate='<b>%{y}</b><br>%{x:.0f}% of this area<extra></extra>')
        fig.add_scatter(x=cmp['all'], y=cmp.index, mode='markers', name='All postings',
                        marker=dict(symbol='line-ns', size=16, line=dict(width=2.5, color='#898781')),
                        hovertemplate='<b>%{y}</b><br>%{x:.0f}% of all postings<extra></extra>')
        fig.update_xaxes(ticksuffix='%', title='Share of postings')
        fig.update_layout(legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(layout(fig, 'Seniority mix', 'Entry-level or experienced? Grey mark = all postings.',
                               height=max(300, 28 * len(cmp) + 130)), use_container_width=True)

    # ---- What to bundle with ----
    st.subheader('What these jobs also ask for')
    other = long[long['job_id'].isin(ids) & (long['area'] != focus)]
    left, right = st.columns(2)
    with left:
        o = other.groupby('skill').agg(postings=('job_id', 'nunique'), area=('area', 'first')).sort_values('postings').tail(10)
        if o.empty:
            st.info('These postings rarely list skills from other areas.')
        else:
            o['share'] = o['postings'] / len(ids) * 100
            fig = go.Figure(go.Bar(x=o['share'], y=o.index, orientation='h', marker_color=BLUE, customdata=o[['postings', 'area']],
                                   text=[f'{v:.0f}%' for v in o['share']], textposition='outside', cliponaxis=False,
                                   hovertemplate='<b>%{y}</b> (%{customdata[1]})<br>%{customdata[0]} postings<extra></extra>'))
            fig.update_xaxes(ticksuffix='%', title=f'Share of {focus} postings', range=[0, o['share'].max() * 1.2])
            st.plotly_chart(layout(fig, 'Skills from other areas', 'Candidates to bundle into the same course.',
                                   height=max(300, 28 * len(o) + 110)), use_container_width=True)
    with right:
        oa = other.groupby('area')['job_id'].nunique().sort_values().tail(8)
        if not oa.empty:
            share = oa / len(ids) * 100
            fig = go.Figure(go.Bar(x=share.values, y=share.index, orientation='h', marker_color=BLUE, customdata=oa.values,
                                   text=[f'{v:.0f}%' for v in share.values], textposition='outside', cliponaxis=False,
                                   hovertemplate='<b>%{y}</b><br>%{customdata} postings<extra></extra>'))
            fig.update_xaxes(ticksuffix='%', title=f'Share of {focus} postings', range=[0, share.max() * 1.2])
            st.plotly_chart(layout(fig, 'Overlap with other course areas',
                                   'Postings that also ask for a skill from the other area.',
                                   height=max(300, 28 * len(share) + 110)), use_container_width=True)

    # ---- Example postings ----
    st.subheader('Example postings')
    ex = jobs.sort_values('salary_annual', ascending=False)
    mine = long[long['area'] == focus].groupby('job_id')['skill'].apply(', '.join)
    st.dataframe(pd.DataFrame({
        'Title': ex['title'], 'Company': ex['company'], 'Seniority': ex['seniority_label'], 'State': ex['state'],
        'Posted pay': [('' if u == 'unknown' else f'${v:,.0f} / year' if u == 'annual' else f'${v:,.2f} / hour')
                       for v, u in zip(ex['minSalary'], ex['salary_unit'])],
        f'Skills in {focus}': mine.reindex(ex.index).values}), use_container_width=True, hide_index=True, height=320)
    st.caption(f'{len(jobs):,} postings ask for at least one skill in this area. Follows the global settings in the sidebar.')


def skill_tab():
    sig, skills = load_signals(mode, without_devjobs)
    st.markdown('Pick a skill to see what employers ask for alongside it. Use this to decide what belongs in one course.')
    options = sorted(skills.index)
    c = st.columns([2, 1])
    pick = c[0].selectbox('Skill', options, index=options.index('Python') if 'Python' in options else 0)
    min_co = c[1].slider('Minimum postings together', 2, 15, 5)
    p, n = load_partners(pick, mode, without_devjobs)
    p = p[p['together'] >= min_co]
    if p.empty:
        st.info('No skill appears with this one often enough. Lower the minimum.')
        return

    top = p.head(3)
    st.markdown(f'**Of the {n} postings that mention {pick}**, ' + ', '.join(
        f'{r["share_pct"]:.0f}% also mention {name}' for name, r in top.iterrows()) + '.')

    solid = p['together'] >= 10   # pairs seen in fewer than 10 postings are faded
    g = p.reset_index()
    g['label'] = [s if (ok and (i < 6 or l >= p.loc[solid, 'lift'].quantile(0.8))) else ''
                  for i, (s, ok, l) in enumerate(zip(g['skill'], solid.values, g['lift']))]
    fig = go.Figure(go.Scatter(
        x=g['together'], y=g['lift'], mode='markers+text', text=g['label'], textposition='top center',
        marker=dict(size=11, color=BLUE, opacity=[0.9 if ok else 0.25 for ok in solid.values],
                    line=dict(color='white', width=1.5)),
        customdata=g[['skill', 'share_pct', 'companies', 'area']],
        hovertemplate='<b>%{customdata[0]}</b> (%{customdata[3]})<br>%{x} postings together, %{customdata[2]} companies<br>'
                      f'%{{customdata[1]:.0f}}% of {pick} postings also mention it<br>Lift %{{y:.1f}}<extra></extra>'))
    fig.add_hline(y=1, line_width=1, line_color='#898781')
    fig.update_xaxes(title=f'Postings mentioning both {pick} and the skill')
    layout(fig, f'Skills posted with {pick}: how common and how distinctive',
           'Right = common pairing. Up = distinctive pairing (lift above 1 means together more often than chance). '
           'Faded points have fewer than 10 postings.', height=460)
    fig.update_layout(yaxis_title='Lift')
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame({
        'Course area': p['area'], 'Postings together': p['together'].astype(int), 'Companies': p['companies'].astype(int),
        f'Share of {pick} postings': p['share_pct'].map('{:.0f}%'.format), 'Lift': p['lift'].round(1),
        'Sample': ['ok' if ok else 'small' for ok in solid.values]}).rename_axis('Skill'), use_container_width=True)
    st.caption('A pairing with high lift but few postings can come from a handful of ads. Prefer pairings that are '
               'both common and distinctive.')


def salary_tab():
    with_salary = df.dropna(subset=['salary_annual'])
    st.caption(f'{len(with_salary):,} of the {len(df):,} postings are used here. The other {len(df) - len(with_salary)} '
               'are left out because the posted amount could not be read as hourly or annual pay (for example 250 or '
               '4,000). Salaries are the posted minimum.')
    with st.container(border=True):
        st.markdown('**Filters for this tab**')
        f = st.columns(4)
        industries = f[0].multiselect('Industry', sorted(df['industry'].unique()))
        sources = f[1].multiselect('Job board', sorted(df['source'].unique()))
        levels = f[2].multiselect('Seniority', ['Intern', 'Junior', 'Medior', 'Senior', 'Lead', 'Director', 'Chief'])
        states = f[3].multiselect('State', sorted(df['state'].dropna().unique()))
    d = with_salary
    for col, sel in [('industry', industries), ('source', sources), ('seniority_label', levels), ('state', states)]:
        if sel:
            d = d[d[col].isin(sel)]
    if d.empty:
        st.warning('No postings match the current filters.')
        return

    k = st.columns(4)
    k[0].metric('Postings', f'{len(d):,}')
    k[1].metric('Companies', f"{d['company'].nunique():,}")
    sal, hr = d[d['salary_unit'] == 'annual'], d[d['salary_unit'] == 'hourly']
    k[2].metric('Median salary (salaried)', money(sal['minSalary'].median()) if len(sal) else 'n/a', help=f'{len(sal)} postings')
    k[3].metric('Median hourly wage', f"${hr['minSalary'].median():,.2f}" if len(hr) else 'n/a', help=f'{len(hr)} postings')

    st.subheader('Salary ranges')
    c = st.columns([1, 1, 1])
    pay = c[0].radio('Pay type', ['Salaried', 'Hourly'], horizontal=True,
                     help='Salaried and hourly jobs are different populations, so they are shown separately.')
    dims = {'Seniority': 'seniority_label', 'Industry': 'industry', 'Job board': 'source', 'State': 'state'}
    dim = dims[c[1].selectbox('Group by', list(dims))]
    min_n = c[2].slider('Minimum postings per group', 3, 30, 8)
    part = sal if pay == 'Salaried' else hr
    g = part.groupby(dim)['minSalary'].agg(n='size', q25=lambda s: s.quantile(.25), median='median',
                                           q75=lambda s: s.quantile(.75))
    g = g[g['n'] >= min_n].sort_values('median')
    if g.empty:
        st.info(f'No group has at least {min_n} {pay.lower()} postings.')
    else:
        labels = [f'{a}  (n={n})' for a, n in zip(g.index, g['n'])]
        fmt = ',.0f' if pay == 'Salaried' else ',.2f'
        fig = go.Figure(go.Scatter(
            x=g['median'], y=labels, mode='markers', marker=dict(color=BLUE, size=11),
            error_x=dict(type='data', symmetric=False, array=g['q75'] - g['median'], arrayminus=g['median'] - g['q25'],
                         color=BLUE, thickness=2, width=0),
            customdata=g[['q25', 'q75', 'n']],
            hovertemplate=f'<b>%{{y}}</b><br>Median $%{{x:{fmt}}}<br>Middle half: $%{{customdata[0]:{fmt}}} to '
                          f'$%{{customdata[1]:{fmt}}}<extra></extra>'))
        fig.update_xaxes(tickprefix='$', tickformat=',.0f', rangemode='tozero',
                         title='Posted minimum salary (USD per year)' if pay == 'Salaried' else 'Posted minimum wage (USD per hour)')
        st.plotly_chart(layout(fig, f'{pay} pay by {dim.replace("_label", "").replace("source", "job board")}',
                               'Dot = median. Line = middle half of postings (25th to 75th percentile). '
                               'A long line means pay varies widely within the group.',
                               height=30 * len(g) + 130), use_container_width=True)

    fig = px.histogram(part, x='minSalary', nbins=30, color_discrete_sequence=[BLUE])
    fig.update_traces(marker_line=dict(color='white', width=1), hovertemplate='$%{x}<br>%{y} postings<extra></extra>')
    fig.update_xaxes(tickprefix='$', tickformat=',.0f', title='USD per year' if pay == 'Salaried' else 'USD per hour')
    fig.update_yaxes(title='Postings')
    layout(fig, f'Distribution of {pay.lower()} pay ({len(part):,} postings)', height=320)
    fig.update_layout(yaxis_title='Postings')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Postings')
    show = d.sort_values('salary_annual', ascending=False)
    st.dataframe(pd.DataFrame({
        'Title': show['title'], 'Company': show['company'], 'Industry': show['industry'], 'State': show['state'],
        'Seniority': show['seniority_label'],
        'Posted pay': [f'${v:,.0f} / year' if u == 'annual' else f'${v:,.2f} / hour'
                       for v, u in zip(show['minSalary'], show['salary_unit'])],
        'Job board': show['source'].str.replace('_us', '')}), use_container_width=True, hide_index=True)


def model_card(name, kind, question):
    """Header shown above every model on the Skill ML Insights tab."""
    st.subheader(question)
    st.markdown(f'**Model: {name}** · {kind}')


def takeaways(found, strong, weak):
    """Plain-language notes shown under every model."""
    st.markdown(f'**What it found.** {found}\n\n**Strong point.** {strong}\n\n**Weak point.** {weak}')


def ml_tab():
    r = load_ml(mode, without_devjobs)
    B, rules = r['B'], r['rules']
    st.markdown('Seven machine learning models, all applied to the skills in the job postings. They answer two '
                'questions: **which skills belong together in one course**, and **what should a learner pick up next?** '
                'Each model is shown with what it found, where it is strong and where it is weak. '
                'The summary at the bottom says which models fit skill data best.')
    st.caption(f'Based on the {len(B):,} postings that list at least two skills ({B.shape[1]} skills). '
               'Follows the global settings in the sidebar.')

    strong = rules[(rules['size'] == 1) & (rules['confidence'] >= 0.6) & (rules['lift'] > 1.5)]
    mixed = (r['theta'].max(axis=1) < 0.7).mean()
    n_comm = int((r['label'].value_counts() >= 2).sum())
    k = st.columns(4)
    k[0].metric('Strong skill pairings', f'{len(strong)}', help='From the association rules model: pairs where 60% or more of the postings with one skill also ask for the other')
    k[1].metric('Skill groups found', f'{n_comm}', help='From the network model (Louvain)')
    k[2].metric('Postings mixing job types', f'{mixed:.0%}', help='From the topic model (LDA)')
    k[3].metric('Next-skill hit rate', f"{r['hit']:.0%}", delta=f"{(r['hit'] - r['hit_pop']) * 100:+.0f} pts vs guessing",
                help='From the recommender: how often a hidden skill appears in its top 5 suggestions')

    # ---- 1. Association rules ----
    st.divider()
    model_card('Association rules (Apriori)', 'pattern mining', '1. If a posting asks for one skill, what else does it ask for?')
    c = st.columns([2, 1, 1])
    pick = c[0].multiselect('Starting skills (leave empty to see the strongest rules overall)', sorted(B.columns))
    min_conf = c[1].slider('Minimum confidence', 0.3, 1.0, 0.6, 0.05)
    min_sup = c[2].slider('Minimum postings', 8, 30, 10)
    f = rules[(rules['confidence'] >= min_conf) & (rules['postings'] >= min_sup) & (rules['lift'] > 1)]
    if pick:
        f = f[f['antecedent'].apply(lambda x: set(x.split(' + ')) <= set(pick))]
    f = f.sort_values(['confidence', 'postings'], ascending=False).head(40)
    if f.empty:
        st.info('No rule meets these thresholds. Lower the minimums or change the starting skills.')
    else:
        st.dataframe(pd.DataFrame({'If a posting asks for': f['antecedent'], 'it also asks for': f['consequent'],
                                   'Postings': f['postings'], 'Confidence': f['confidence'].map('{:.0%}'.format),
                                   'Lift': f['lift'].round(1)}), use_container_width=True, hide_index=True)
        st.caption('Confidence = of the postings with the starting skills, the share that also list the other skill. '
                   'Lift above 1 = more often than chance.')
    top = strong.sort_values(['postings', 'confidence'], ascending=False).head(3)
    takeaways(
        (f'{len(strong)} strong pairings. For example: ' + '; '.join(
            f'{x.confidence:.0%} of {x.antecedent} postings also ask for {x.consequent}' for x in top.itertuples()) + '.')
        if len(top) else 'No strong pairings with the current settings.',
        'The easiest result to act on. Every rule is a plain statement with a count behind it, and it can combine two '
        'starting skills to find three-way patterns.',
        'It only sees skills that appear often. A rule based on fewer than about 15 postings can be a coincidence.')

    # ---- 2. Recommender ----
    st.divider()
    model_card('Skill recommender (item-based collaborative filtering)', 'recommendation',
               '2. Given the skills someone has, what should they learn next?')
    have = st.multiselect('Skills a learner already has', sorted(B.columns),
                          default=[s for s in ['Python', 'SQL'] if s in B.columns])
    rec = ml.recommend(have, r['S'], B.columns, 8)
    if rec.empty:
        st.info('Pick at least one skill.')
    else:
        g = rec.sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation='h', marker_color=BLUE,
                               customdata=[sa.AREA_OF[s] for s in g.index],
                               hovertemplate='<b>%{y}</b> (%{customdata})<br>Score %{x:.2f}<extra></extra>'))
        fig.update_xaxes(title='How closely the skill goes with the ones already held')
        st.plotly_chart(layout(fig, 'Recommended next skills', height=330), use_container_width=True)
    takeaways(
        f"When one skill is hidden from a posting, the model puts it in its top 5 suggestions {r['hit']:.0%} of the "
        f"time. Simply suggesting the most popular skills gets {r['hit_pop']:.0%}.",
        'It was tested on postings it had not seen, so the score is an honest measure. It turns directly into a '
        '"what to learn next" path for learners.',
        'It follows what employers already ask for together, so it will not suggest a new skill that is only starting '
        'to appear.')

    # ---- 3. Louvain ----
    st.divider()
    model_card('Network community detection (Louvain)', 'unsupervised grouping',
               '3. Which skills form natural groups?')
    label, strength = r['label'], r['strength']
    rows = []
    for c_id, members in label.groupby(label):
        if len(members) < 2:
            continue
        names = sorted(members.index, key=lambda s: -strength[s])
        areas = pd.Series([sa.AREA_OF[s] for s in names]).value_counts()
        has = B[names].sum(axis=1) >= 2
        rows.append({'Group': int(c_id), 'Skills (most connected first)': ', '.join(names),
                     'Course areas it spans': ', '.join(f'{x} ({n})' for x, n in areas.items()),
                     'Postings': int(has.sum()), 'Median salary': money(r['salary'][has.values].median())})
    st.dataframe(pd.DataFrame(rows).set_index('Group'), use_container_width=True)
    g = r['between'].sort_values().tail(8)
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation='h', marker_color=BLUE,
                           hovertemplate='<b>%{y}</b><br>Bridge score %{x:.3f}<extra></extra>'))
    fig.update_xaxes(title='Bridge score (betweenness centrality)')
    st.plotly_chart(layout(fig, 'Bridge skills', 'Skills that connect different groups. Good candidates for foundation '
                           'content shared by several courses.', height=330), use_container_width=True)
    bridges = ', '.join(r['between'].sort_values(ascending=False).head(3).index)
    takeaways(
        f'{n_comm} groups of skills that employers ask for together. The skills that link the groups most are {bridges}.',
        'Nobody tells it what the groups should be, so it is an independent check on the course areas defined by hand. '
        'It also shows which skills connect several groups.',
        f"The groups overlap. The separation score (modularity) is {r['modularity']:.2f}, where 0.3 or more counts as "
        'clear structure, and most tech skills end up in a few large groups.')

    # ---- 4. Hierarchical ----
    st.divider()
    model_card('Hierarchical clustering', 'unsupervised grouping', '4. Which skills are almost always asked for together?')
    hier = r['groups']['Hierarchical clustering']
    rows = []
    for _, members in hier.groupby(hier):
        if len(members) >= 2:
            names = sorted(members.index, key=lambda s: -B[s].sum())
            rows.append({'Skills in the bundle': ', '.join(names), 'Skills': len(names),
                         'Postings with 2+ of them': int((B[names].sum(axis=1) >= 2).sum())})
    hb = pd.DataFrame(rows).sort_values('Postings with 2+ of them', ascending=False)
    st.dataframe(hb, use_container_width=True, hide_index=True)
    takeaways(
        f'{len(hb)} tight bundles. The largest by postings is: {hb.iloc[0, 0]}.' if len(hb) else 'No tight bundles with the current settings.',
        'It gives small, precise bundles, each about the size of one course module.',
        f'It leaves {B.shape[1] - (int(hb["Skills"].sum()) if len(hb) else 0)} of the {B.shape[1]} skills on their own, '
        'and the cut-off that decides how tight a bundle must be is chosen by the analyst.')

    # ---- 5. LDA ----
    st.divider()
    model_card('Topic model (LDA)', 'unsupervised grouping', '5. What types of jobs are in the market?')
    phi = r['phi']
    main = r['theta'].idxmax(axis=1)
    st.dataframe(pd.DataFrame({
        'Typical skills': [', '.join(phi.loc[i].sort_values(ascending=False).head(6).index) for i in phi.index],
        'Postings': [int((main == i).sum()) for i in phi.index],
        'Median salary': [money(r['salary'][(main == i).values].median()) for i in phi.index]},
        index=pd.Index(phi.index, name='Job type')), use_container_width=True)
    takeaways(
        f'{len(phi)} job types, from Office-based roles to cloud and front-end roles. {mixed:.0%} of postings are a '
        'mix of two or more types.',
        'It allows one posting to belong to several types at once, which matches real jobs. That mix is the evidence '
        'for building courses that combine two bundles.',
        'The number of job types is chosen by the analyst, and the types blur into each other at the edges.')

    # ---- 6. Embeddings ----
    st.divider()
    model_card('Skill embeddings (Word2Vec-style)', 'unsupervised similarity', '6. Which skills are similar to each other?')
    sim = r['sim']
    options = sorted(sim.columns)
    probe = st.selectbox('Skill', options, index=options.index('React') if 'React' in options else 0, key='emb_skill')
    s = sim[probe].drop(probe).sort_values(ascending=False).head(8)[::-1]
    together = [int((B[probe] & B[x]).sum()) for x in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h', customdata=together,
                           marker_color=[BLUE if t >= 5 else GRAY for t in together],
                           hovertemplate='<b>%{y}</b><br>Similarity %{x:.2f}<br>Posted together %{customdata} times<extra></extra>'))
    fig.update_xaxes(title='Similarity (0 to 1)', range=[0, 1])
    st.plotly_chart(layout(fig, f'Skills most similar to {probe}',
                           'Grey bars: posted together fewer than 5 times, so the similarity has little evidence behind it.',
                           height=330), use_container_width=True)
    takeaways(
        'For skills with many postings the neighbours make sense (React sits next to Angular, JavaScript and '
        'TypeScript; Excel next to PowerPoint and Word). For less common skills the neighbours are often unrelated.',
        'In principle it can spot two skills that are alternatives to each other even if they are never posted together.',
        'It needs far more data than this. With a few hundred postings many similarities rest on a handful of ads, '
        'which is why some bars are grey.')

    # ---- 7. Title classifier ----
    st.divider()
    model_card('Skill prediction from job title (logistic regression)', 'supervised classification',
               '7. Can a job title tell us which skills the job needs?')
    st.markdown('**How it works.** The model reads only the job title, for example "Senior Java Developer", and guesses '
                'which skills the posting asks for. Its guesses are then checked against postings where the skills are '
                'known. If this worked well, it could fill in skills for the many postings that list none.')
    sc = r['title_scores'].sort_values('average_precision')
    fig = go.Figure()
    fig.add_bar(x=sc['average_precision'], y=sc.index, orientation='h', marker_color=BLUE, name='Model',
                customdata=sc[['postings', 'base_rate']],
                hovertemplate='<b>%{y}</b><br>Model score %{x:.2f}<br>Random guessing %{customdata[1]:.2f}<br>'
                              '%{customdata[0]} postings<extra></extra>')
    fig.add_scatter(x=sc['base_rate'], y=sc.index, mode='markers', name='Random guessing',
                    marker=dict(symbol='line-ns', size=14, line=dict(width=2.5, color='#898781')), hoverinfo='skip')
    fig.update_xaxes(title='Prediction quality (average precision, 0 to 1)', range=[0, 1])
    fig.update_layout(legend=dict(orientation='h', y=-0.12))
    st.plotly_chart(layout(fig, 'How well the title predicts each skill',
                           'Longer bar = better guesses (1 would be perfect). Grey mark = the score random guessing '
                           'would get. A bar that barely passes its grey mark means the model is close to guessing.',
                           height=26 * len(sc) + 150), use_container_width=True)
    sg = r['title_sugg']
    st.markdown('**What it suggests for postings that have no skill listed** (its most confident guesses)')
    st.dataframe(pd.DataFrame({'Job title': sg['title'], 'Suggested skill': sg['suggested_skill'],
                               'Confidence': sg['probability'].map('{:.0%}'.format)}).head(12),
                 use_container_width=True, hide_index=True)
    best, worst = sc.index[-1], sc.index[0]
    takeaways(
        f'It guesses well only when the skill is usually named in the job title. {best} is the best case: the word '
        f'"{best}" often appears in the title itself, and the model scores {sc["average_precision"].iloc[-1]:.2f} out of 1 '
        f'(random guessing would score {sc["base_rate"].iloc[-1]:.2f}). {worst} is the worst case: the jobs that need it '
        f'have all kinds of titles that never mention it, so the model scores {sc["average_precision"].iloc[0]:.2f}, '
        'barely better than guessing. The table above shows the result: sensible guesses mixed with clearly wrong ones.',
        'It is the only model here that could fill in skills for the many postings that list none.',
        'A title is too little to go on. Even its most confident guesses include mistakes, so it should not be used to '
        'fill in data without the full job description.')

    # ---- Agreement ----
    st.subheader('Do the grouping models agree with each other?')
    st.markdown('**Compared: manual course areas, Louvain, LDA and hierarchical clustering**')
    left, right = st.columns([1, 1])
    a = r['ari'].round(2)
    short = {'Manual course areas': 'Manual areas', 'Network communities (Louvain)': 'Louvain',
             'Topic model (LDA)': 'LDA', 'Hierarchical clustering': 'Hierarchical'}
    off = a.values[~np.eye(len(a), dtype=bool)]
    with left:
        fig = go.Figure(go.Heatmap(z=a.values, x=[short[c] for c in a.columns], y=[short[c] for c in a.index],
                                   zmin=0, zmax=1, colorscale=[[0, '#cde2fb'], [1, '#184f95']], showscale=False,
                                   xgap=2, ygap=2, text=a.values, texttemplate='%{text:.2f}',
                                   hovertemplate='%{y} vs %{x}<br>Agreement %{z:.2f}<extra></extra>'))
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(layout(fig, 'Agreement between four ways of grouping skills',
                               '1 = identical groups, 0 = no better than chance (adjusted Rand index).', height=330),
                        use_container_width=True)
    with right:
        st.markdown(f'- Agreement runs from **{off.min():.2f}** to **{off.max():.2f}**: better than chance, but low to moderate.\n'
                    '- The models agree on the **core** of each bundle (cloud, front-end, Office, networking). '
                    'They disagree on where one bundle ends and the next begins.\n'
                    '- A pairing is solid when several models show it. The edges of any grouping are a judgement call.')

    # ---- Summary ----
    st.divider()
    st.subheader('Summary: which models fit skill data best')
    st.table(pd.DataFrame([
        {'Model': 'Association rules (Apriori)', 'Fit for skills': 'Best', 'Best used for': 'Deciding what goes into a course',
         'What it showed here': f'{len(strong)} pairs of skills that employers usually ask for together'},
        {'Model': 'Skill recommender', 'Fit for skills': 'Best', 'Best used for': 'Learning paths: what to learn next',
         'What it showed here': f"Its top 5 suggestions include the right next skill {r['hit']:.0%} of the time. "
                                f"Just suggesting popular skills manages {r['hit_pop']:.0%}"},
        {'Model': 'Network communities (Louvain)', 'Fit for skills': 'Good', 'Best used for': 'Big-picture groups and bridge skills',
         'What it showed here': f'{n_comm} groups of related skills, though the groups overlap'},
        {'Model': 'Topic model (LDA)', 'Fit for skills': 'Good', 'Best used for': 'Seeing which job types exist and how they mix',
         'What it showed here': f'{len(phi)} job types. {mixed:.0%} of postings combine two or more of them'},
        {'Model': 'Hierarchical clustering', 'Fit for skills': 'Good', 'Best used for': 'Small, tight course modules',
         'What it showed here': f'{len(hb)} small bundles of skills that are almost always asked for together'},
        {'Model': 'Skill embeddings', 'Fit for skills': 'Needs more data', 'Best used for': 'Finding similar or alternative skills',
         'What it showed here': 'Sensible results for common skills such as React or Excel. Unreliable for less common skills'},
        {'Model': 'Skill prediction from job title', 'Fit for skills': 'Not reliable yet', 'Best used for': 'Filling in missing skills',
         'What it showed here': f'Guesses well only when the skill is named in the job title (such as {best}). For general '
                                f'skills that titles never mention (such as {worst}) it is close to random guessing'},
    ]).set_index('Model'))
    st.markdown(
        '**The short version**\n'
        '- **Use association rules first.** They give clear statements such as "most AWS postings also ask for Cloud '
        'Computing", which map straight onto course content.\n'
        '- **Use the recommender for learning paths.** It was tested on postings it had not seen and clearly beats '
        'guessing.\n'
        '- **Use the grouping models (Louvain, LDA, hierarchical) together, not alone.** Each draws the lines a little '
        'differently. Where they agree, the bundle is real.\n'
        '- **Hold off on embeddings and title prediction.** Both are promising ideas that need more postings, or the '
        'full job description, before their results can be trusted.')
    st.caption(f'All results are based on {len(B):,} postings, so treat them as strong leads to check, not final answers.')

SCHEMES = {'expert': 'Original weights', 'equal': 'Equal', 'entropy': 'Entropy', 'critic': 'CRITIC', 'pca': 'PCA'}
DIM_NAMES = {'demand': 'Demand', 'salary': 'Salary premium', 'trend': 'Trend', 'breadth': 'Breadth'}


def robustness_tab():
    tables, report = load_robustness(mode, without_devjobs)
    st.markdown('The opportunity score on the first tab uses weights we set ourselves (demand 45%, salary 30%, '
                'trend 10%, breadth 15%). They are a judgement call with no published source, so this page checks '
                'whether the ranking holds up when the weights change, following the OECD & JRC (2008) '
                '*Handbook on Constructing Composite Indicators* and Saisana, Saltelli & Tarantola (2005).')
    level = st.radio('Rank', ['Skills', 'Course areas'], horizontal=True)
    key = 'skill' if level == 'Skills' else 'area'
    t, R = tables[key]
    k = report[key]['top_k']
    p_col = f'dir_flat_p_top{k}'
    stable = t.index[t[p_col] >= 0.8].tolist()
    unit = 'skills' if key == 'skill' else 'course areas'

    st.subheader(f'Stable top {k}')
    st.markdown(f'Ranked again with 10,000 random weight sets. **{len(stable)} of the {len(t)} {unit} stay in the top {k} in at '
                f'least 80% of them:** ' + (', '.join(stable) if stable else 'none') + '.')

    show = t.sort_values('dir_flat_median_rank').head(20 if key == 'skill' else len(t)).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=show['dir_flat_rank_p95'], y=show.index, mode='markers', marker=dict(size=1, color=GRAY),
                             error_x=dict(type='data', symmetric=False, array=[0] * len(show),
                                          arrayminus=show['dir_flat_rank_p95'] - show['dir_flat_rank_p05'],
                                          color=GRAY, thickness=6, width=0),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=show['dir_flat_median_rank'], y=show.index, mode='markers',
                             marker=dict(size=11, color=BLUE, line=dict(color='white', width=1.5)),
                             customdata=show[[p_col, 'rank_expert']].values, showlegend=False,
                             hovertemplate='<b>%{y}</b><br>Median rank %{x:.0f}<br>'
                                           f'In the top {k} in %{{customdata[0]:.0%}} of weight sets<br>'
                                           'Rank with the original weights %{customdata[1]}<extra></extra>'))
    fig.update_xaxes(title='Rank (1 = best)', autorange='reversed')
    layout(fig, 'How much the rank moves when the weights change',
           'Dot = median rank over 10,000 random weight sets. Bar = the middle 90% of ranks.',
           height=max(320, 26 * len(show) + 120))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Rank under different weighting methods')
    st.markdown('Besides random weights, the weights can be set from the data. **Entropy** gives more weight to the '
                'signal that varies most. **CRITIC** (Diakoulaki, Mavrotas & Papayannakis 1995) also discounts '
                'signals that repeat each other. **PCA** weights follow the main pattern in the data. Where all of '
                'them agree, the result is the most credible.')
    w = pd.DataFrame(report[key]['weights']).T.rename(index=SCHEMES, columns=DIM_NAMES)
    st.dataframe(w.style.format('{:.0%}'), use_container_width=True)
    corr = report[key]['indicator_correlation']['demand']['breadth']
    st.caption(f'Postings (demand) and companies (breadth) correlate at {corr:.2f}, so they measure nearly the same '
               'thing. With the original weights demand effectively counts twice; CRITIC corrects for that.')
    cols = {f'rank_{s}': n for s, n in SCHEMES.items()}
    tab = t[list(cols)].rename(columns=cols)
    tab[f'Top {k} share'] = t[p_col].map('{:.0%}'.format)
    tab['Pareto front'] = np.where(t['pareto_layer'] == 1, 'Yes', '')
    tab['Industries with RCA ≥ 1'] = t['industries_rca_ge1']
    if key == 'skill':
        tab.insert(0, 'Course area', t['area'])
    st.dataframe(tab.sort_values('Original weights').rename_axis(level[:-1]), use_container_width=True, height=420)
    st.caption('Industries with RCA ≥ 1: in how many industries (with at least 20 postings) the item appears more '
               'often than in the data as a whole, after Alabdulkareem et al. (2018). Unlike the company count, it '
               'does not simply grow with the number of postings.')

    st.subheader('Pareto front: no weights at all')
    X = t.assign(lbl=np.where((t['pareto_layer'] == 1) | t.index.isin(stable), t.index, ''))
    front = X['pareto_layer'] == 1
    fig = go.Figure()
    for mask, name, color, size in [(~front, 'Other', GRAY, 8), (front, 'On the Pareto front', BLUE, 13)]:
        g = X[mask]
        fig.add_trace(go.Scatter(x=g['jobs'], y=g['salary_premium_pct'], mode='markers+text', name=name,
                                 text=g['lbl'], textposition='top center',
                                 marker=dict(size=size, color=color, line=dict(color='white', width=1)),
                                 customdata=np.column_stack([g['companies'], g['pareto_layer'], g.index]),
                                 hovertemplate='<b>%{customdata[2]}</b><br>%{x} postings, %{customdata[0]} companies<br>'
                                               'Salary difference %{y:.0f}%<br>Layer %{customdata[1]}<extra></extra>'))
    fig.update_xaxes(title='Postings', type='log')
    layout(fig, f'{level} that no other item beats on demand, salary and breadth at once',
           'Up = higher pay after controlling for seniority and job board. Right = more postings.', height=480)
    fig.update_layout(yaxis_title='Salary difference (%)', legend=dict(orientation='h', y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Trend is left out here because most items have too few postings at both time points to measure it.')

    if key == 'skill' and 'onet_check' in report:
        o = report['onet_check']
        st.subheader('Check against an outside source')
        st.markdown(f'{o["labelled_skills"]} skills match an entry in the O*NET 29.3 technology list, which flags '
                    f'{o["hot"]} of them as Hot Technologies. The score ranks a hot skill above a non-hot one '
                    f'{o["auc"]["score_expert"]:.0%} of the time with the original weights '
                    f'(CRITIC {o["auc"]["score_critic"]:.0%}, entropy {o["auc"]["score_entropy"]:.0%}).')
        st.caption('O*NET flags hot technologies by how often they appear in job ads, so this confirms the demand '
                   f'signal rather than proving the score can spot emerging skills. Only {o["not_hot"]} skills are '
                   'not hot, so the figure is rough. Arusto enrollment data would make a stronger check.')


t1, t2, t3, t4, t5, t6 = st.tabs(['Course Opportunities', 'Score Robustness', 'Area Deep Dive', 'Skill Bundles',
                                  'Skill ML Insights', 'Salary Overview'])
with t1:
    course_tab()
with t2:
    robustness_tab()
with t3:
    area_tab()
with t4:
    skill_tab()
with t5:
    ml_tab()
with t6:
    salary_tab()
