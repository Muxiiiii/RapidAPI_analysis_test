"""Arusto job market dashboard. Run: streamlit run app.py
Data is jobs_clean.csv, the cleaned output of jobs_analysis.ipynb."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    return dict(B=B, rules=ml.association_rules(B), label=label, modularity=modularity, between=between,
                strength=strength, phi=phi, theta=theta, ari=ari, groups=groups, hit=hit, hit_pop=hit_pop,
                n_test=n_test, S=ml.item_similarity(B), salary=d.loc[B.index, 'salary_annual'])


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
st.sidebar.caption('Apply to Course Opportunities, Skill Bundles and Skill ML Insights. Salary Overview has its own filters on its tab.')
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

    top = st.container()   # filled after the weights are read, but shown first
    with st.expander('Adjust scoring weights'):
        st.caption('The opportunity score ranks each area on four signals and averages the ranks with these weights.')
        w = st.columns(4)
        wd = w[0].slider('Hiring demand', 0.0, 1.0, 0.45, 0.05, help='Number of postings asking for skills in the area')
        ws = w[1].slider('Salary difference', 0.0, 1.0, 0.30, 0.05,
                         help='Pay of postings in the area compared with other postings of the same seniority and job board')
        wb = w[2].slider('Employer coverage', 0.0, 1.0, 0.15, 0.05, help='Number of distinct companies hiring')
        wt = w[3].slider('Two-period change', 0.0, 1.0, 0.10, 0.05,
                         help='Change in share of postings between Sep 2025 and Mar 2026. Neutral where the sample is too small.')
    weights = (wd, ws, wt, wb)
    full = area_table(sig_all, sig_wo, mix, weights)
    # the view follows the global devjobs switch; `full` always keeps both rankings for comparison
    area = area_table(sig_wo, sig_wo, mix, weights) if without_devjobs else full
    if without_devjobs:
        area['devjobs_pct'] = full['devjobs_pct'].reindex(area.index)
        area['confidence'] = [confidence(j, 0) for j in area['jobs']]

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

        focus = st.selectbox('Focus area', area.index.tolist(), help='Highlighted in the charts below')

        left, right = st.columns(2)
        with left:
            g = area.sort_values('score')
            fig = go.Figure(go.Bar(x=g['score'], y=g.index, orientation='h', text=g['score'].round().astype(int),
                                   textposition='outside', cliponaxis=False,
                                   marker_color=[BLUE if a == focus else GRAY for a in g.index],
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
            fig.update_xaxes(side='top', tickangle=0)
            st.plotly_chart(layout(fig, 'What drives each score',
                                   'Rank on each signal: 0 = weakest area, 100 = strongest.',
                                   height=30 * len(h) + 110), use_container_width=True)

        # Demand vs salary difference: label only the focus area and the top three
        g = area.reset_index()
        g['label'] = [a if (a == focus or i < 3) else '' for i, a in enumerate(g['area'])]
        g['pos'] = ['bottom center' if (i == 1 and a != focus) else 'top center' for i, a in enumerate(g['area'])]
        fig = go.Figure(go.Scatter(
            x=g['jobs'], y=g['salary_premium_pct'], mode='markers+text', text=g['label'], textposition=g['pos'],
            marker=dict(size=g['companies'], sizemode='area', sizeref=2 * g['companies'].max() / 44 ** 2, sizemin=5,
                        color=[BLUE if a == focus else GRAY for a in g['area']], line=dict(color='white', width=2)),
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

        st.subheader(f'Skills within {focus}')
        sk = sa.opportunity(skill_wo if without_devjobs else skill_all, *weights)
        sk = sk[sk['area'] == focus].sort_values('jobs', ascending=False)
        if sk.empty:
            st.info('No skill in this area has at least 5 postings.')
        else:
            left, right = st.columns(2)
            with left:
                g = sk.head(15).sort_values('jobs')
                fig = go.Figure(go.Bar(x=g['jobs'], y=g.index, orientation='h', marker_color=BLUE, text=g['jobs'],
                                       textposition='outside', cliponaxis=False,
                                       hovertemplate='<b>%{y}</b><br>%{x} postings<extra></extra>'))
                fig.update_xaxes(title='Postings')
                st.plotly_chart(layout(fig, 'Most requested skills', height=28 * len(g) + 100), use_container_width=True)
            with right:
                st.dataframe(pd.DataFrame({
                    'Postings': sk['jobs'].astype(int), 'Companies': sk['companies'].astype(int),
                    'Median salary': sk['median_salary'].map(money),
                    'Salary difference': [f'{v:+.0f}%' if n >= 20 else 'too few postings'
                                          for v, n in zip(sk['salary_premium_pct'], sk['jobs'])]}).rename_axis('Skill'),
                    use_container_width=True)
            st.caption('Salary difference is shown only for skills with at least 20 postings.')

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


def ml_tab():
    r = load_ml(mode, without_devjobs)
    B, rules = r['B'], r['rules']
    st.markdown('Six machine learning methods applied to the skills in the postings. Each one answers a version of the '
                'same question: **which skills belong together in one course, and what should a learner pick up next?**')
    st.caption(f'Based on the {len(B):,} postings that list at least two curated skills ({B.shape[1]} skills). '
               'Follows the global settings in the sidebar. Full walk-through in skills_ml_experiments.ipynb.')

    # ---- Headline ----
    st.subheader('What the models found')
    strong = rules[(rules['size'] == 1) & (rules['confidence'] >= 0.6) & (rules['lift'] > 1.5)]
    mixed = (r['theta'].max(axis=1) < 0.7).mean()
    k = st.columns(4)
    k[0].metric('Strong skill pairings', f'{len(strong)}', help='Rules A → C with confidence of 60% or more and lift above 1.5')
    k[1].metric('Skill communities', f"{(r['label'].value_counts() >= 2).sum()}",
                help=f"Found by the Louvain algorithm. Modularity {r['modularity']:.2f}: above 0.3 means clear structure.")
    k[2].metric('Postings mixing topics', f'{mixed:.0%}', help='Share of postings whose main LDA topic covers less than 70% of their skills')
    k[3].metric('Recommender hit rate', f"{r['hit']:.0%}", delta=f"{(r['hit'] - r['hit_pop']) * 100:+.0f} pts vs popular skills",
                help='A hidden skill is found in the top 5 recommendations. Baseline: always recommend the most popular skills.')
    top = strong.sort_values(['postings', 'confidence'], ascending=False).head(5)
    if len(top):
        st.markdown('**Most reliable pairings:** ' + '; '.join(
            f'{x.confidence:.0%} of {x.antecedent} postings also ask for {x.consequent}' for x in top.itertuples()) + '.')

    # ---- Do the methods agree? ----
    st.subheader('Do different methods find the same skill groups?')
    left, right = st.columns([1, 1])
    with left:
        a = r['ari'].round(2)
        short = {'Manual course areas': 'Manual areas', 'Network communities (Louvain)': 'Louvain',
                 'Topic model (LDA)': 'LDA', 'Hierarchical clustering': 'Hierarchical'}
        fig = go.Figure(go.Heatmap(z=a.values, x=[short[c] for c in a.columns], y=[short[c] for c in a.index],
                                   zmin=0, zmax=1, colorscale=[[0, '#cde2fb'], [1, '#184f95']], showscale=False,
                                   xgap=2, ygap=2, text=a.values, texttemplate='%{text:.2f}',
                                   hovertemplate='%{y} vs %{x}<br>Agreement %{z:.2f}<extra></extra>'))
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(layout(fig, 'Agreement between four ways of grouping skills',
                               'Adjusted Rand index: 1 = identical groups, 0 = no better than chance.', height=330),
                        use_container_width=True)
    with right:
        st.markdown('**How to read this.** The manual course areas were defined by hand. The other three groupings '
                    'come from algorithms that only see which skills are posted together.')
        off = a.values[~np.eye(len(a), dtype=bool)]
        st.markdown(f'- Agreement ranges from **{off.min():.2f}** to **{off.max():.2f}**: better than chance, but '
                    'low to moderate. The methods agree on the **core** of each bundle (cloud, front-end, Office, '
                    'networking) and disagree on where the boundaries fall.\n'
                    '- The algorithms merge areas that the manual list keeps apart when employers ask for them '
                    'together, and split areas whose skills are rarely posted together.\n'
                    '- Treat a pairing as solid only when it shows up under several methods, and treat the edges of '
                    'any grouping as a judgement call.')

    # ---- Communities ----
    st.subheader('Skill communities found by the network model')
    label, strength = r['label'], r['strength']
    rows = []
    for c, members in label.groupby(label):
        if len(members) < 2:
            continue
        names = sorted(members.index, key=lambda s: -strength[s])
        areas = pd.Series([sa.AREA_OF[s] for s in names]).value_counts()
        has = B[names].sum(axis=1) >= 2
        rows.append({'Community': int(c), 'Skills (most connected first)': ', '.join(names),
                     'Course areas it spans': ', '.join(f'{x} ({n})' for x, n in areas.items()),
                     'Postings': int(has.sum()), 'Median salary': money(r['salary'][has.values].median())})
    st.dataframe(pd.DataFrame(rows).set_index('Community'), use_container_width=True)
    st.caption('Skills are linked when posted together at least 4 times and at least twice as often as chance. '
               'A community that spans several course areas suggests those areas could be taught as one course.')

    left, right = st.columns(2)
    with left:
        g = r['between'].sort_values().tail(10)
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation='h', marker_color=BLUE,
                               hovertemplate='<b>%{y}</b><br>Betweenness %{x:.3f}<extra></extra>'))
        fig.update_xaxes(title='Betweenness centrality')
        st.plotly_chart(layout(fig, 'Bridge skills', 'Skills that connect otherwise separate bundles: candidates for '
                               'shared foundation content.', height=380), use_container_width=True)
    with right:
        phi = r['phi']
        main = r['theta'].idxmax(axis=1)
        t = pd.DataFrame({'Top skills': [', '.join(phi.loc[i].sort_values(ascending=False).head(5).index) for i in phi.index],
                          'Postings': [int((main == i).sum()) for i in phi.index],
                          'Median salary': [money(r['salary'][(main == i).values].median()) for i in phi.index]},
                         index=pd.Index(phi.index, name='Topic'))
        st.markdown('**Topics found by LDA**')
        st.dataframe(t, use_container_width=True)
        st.caption(f'{mixed:.0%} of postings mix several topics, so courses that combine two bundles match how '
                   'employers hire.')

    # ---- Rules ----
    st.subheader('Association rules: if a posting asks for A, what else does it ask for?')
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
        st.caption('Confidence: of the postings with the starting skills, the share that also list the other skill. '
                   'Lift above 1 means more often than chance. Rules with two starting skills capture three-way '
                   'patterns that pairwise comparison misses.')

    # ---- Recommender ----
    st.subheader('What to learn next')
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
        fig.update_xaxes(title='Similarity to the skills already held')
        st.plotly_chart(layout(fig, 'Recommended next skills', height=330), use_container_width=True)
    st.caption(f"Tested by hiding one skill from each of {r['n_test']} postings: the recommender finds it in its top 5 "
               f"{r['hit']:.0%} of the time, against {r['hit_pop']:.0%} for always recommending the most popular skills.")

    # ---- Method comparison ----
    st.subheader('How the methods compare')
    best = r['ari'].drop(index='Manual course areas')['Manual course areas']
    st.dataframe(pd.DataFrame([
        {'Method': 'Association rules', 'Type': 'Pattern mining', 'Question it answers': 'If A, then what else?',
         'Result on this data': f'{len(strong)} strong pairings', 'Verdict': 'Most directly usable'},
        {'Method': 'Network communities (Louvain)', 'Type': 'Unsupervised',
         'Question it answers': 'Which skills form a group?',
         'Result on this data': f"Modularity {r['modularity']:.2f}; agreement with manual areas {best['Network communities (Louvain)']:.2f}",
         'Verdict': 'Useful, groups overlap'},
        {'Method': 'Hierarchical clustering', 'Type': 'Unsupervised', 'Question it answers': 'Which skills form a group?',
         'Result on this data': f"Agreement with manual areas {best['Hierarchical clustering']:.2f}",
         'Verdict': 'Useful for small, tight bundles'},
        {'Method': 'Topic model (LDA)', 'Type': 'Unsupervised', 'Question it answers': 'Which job types exist?',
         'Result on this data': f'{mixed:.0%} of postings mix topics', 'Verdict': 'Useful, shows postings are mixed'},
        {'Method': 'Skill recommender', 'Type': 'Collaborative filtering', 'Question it answers': 'What to learn next?',
         'Result on this data': f"Top-5 hit rate {r['hit']:.0%} vs {r['hit_pop']:.0%} baseline", 'Verdict': 'Works'},
        {'Method': 'Skill embeddings', 'Type': 'Unsupervised', 'Question it answers': 'Which skills are similar?',
         'Result on this data': 'Clear only for dense bundles', 'Verdict': 'Needs more data (notebook only)'},
        {'Method': 'Skills from job title', 'Type': 'Supervised classification', 'Question it answers': 'Can missing tags be filled in?',
         'Result on this data': 'Good for Java, Networking, Cloud; weak for Office skills', 'Verdict': 'Not reliable enough (notebook only)'},
    ]), use_container_width=True, hide_index=True)
    st.caption('The last two methods are documented in skills_ml_experiments.ipynb and not shown here because their '
               'results were weak. Salary models (decision tree, random forest, XGBoost with SHAP) are in '
               'ml_experiments.ipynb. All results here rest on a few hundred postings and should be read as leads.')


t1, t2, t3, t4 = st.tabs(['Course Opportunities', 'Skill Bundles', 'Skill ML Insights', 'Salary Overview'])
with t1:
    course_tab()
with t2:
    skill_tab()
with t3:
    ml_tab()
with t4:
    salary_tab()
