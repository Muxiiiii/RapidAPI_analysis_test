"""Skill demand analysis: group API skill tags into course areas and compute market signals for each.
Shared by the notebook and the Streamlit app."""
import numpy as np
import pandas as pd

# Course area -> raw skill tags (lowercase). Only real skills are included;
# obvious mislabels and generic words (support, web, ms-dos, soap, ocaml, pizza, ...) are left out,
# and so are job titles (product manager, project manager), which are roles rather than skills.
TAXONOMY = {
    'Cloud & DevOps': ['aws', 'azure', 'gcp', 'cloud', 'cloud computing', 'devops', 'ci / cd', 'kubernetes', 'docker',
                       'terraform', 'jenkins', 'ansible', 'orchestration', 'helm', 'serverless', 'gitlab', 'devsecops',
                       'microservices', 'openstack', 'grafana'],
    'Databases & Data Engineering': ['sql', 'mysql', 'postgresql', 'microsoft sql server', 'oracle database', 'oracle',
                                     'mongodb', 'nosql', 'apache spark', 'apache kafka', 'etl', 'airflow', 'big data',
                                     'bigquery', 'data management'],
    'Data Analytics & BI': ['microsoft excel', 'excel', 'spreadsheet', 'power bi', 'tableau', 'a / b testing', 'matlab'],
    'AI & Machine Learning': ['ai', 'machine-learning', 'machine learning', 'deep learning', 'tensorflow', 'llm'],
    'Programming & Software Engineering': ['python', 'java', 'c++', 'c sharp', 'rust', 'php', 'cobol', 'coding',
                                           'software engineering', 'debugging', 'unit testing', 'code review', 'git',
                                           'github', 'bitbucket', 'dotnet', 'spring', 'maven'],
    'Web Development': ['javascript', 'typescript', 'react.js', 'react', 'html', 'html5', 'css', 'angular', 'angularjs',
                        'node', 'node.js', 'nodejs', 'vue.js', 'vue', 'asp.net', 'jquery', 'laravel', 'rest', 'api',
                        'graphql', 'json', 'backend', 'fullstack'],
    'Cybersecurity': ['security', 'firewall', 'vulnerability', 'iam', 'oauth', 'vpn', 'waf', 'antivirus software', 'jwt'],
    'IT Support & Networking': ['network', 'hardware', 'windows', 'microsoft windows', 'windows server', 'linux', 'unix',
                                'macos', 'powershell', 'bash', 'shell script', 'itil', 'cisco', 'tcp / ip', 'lan',
                                'active directory', 'office 365', 'microsoft 365', 'vmware', 'hyper-v', 'servicenow'],
    'Office Productivity': ['microsoft office', 'microsoft powerpoint', 'microsoft word', 'sharepoint', 'google drive',
                            'slack'],
    'Agile & Project Management': ['jira', 'scrum', 'confluence'],
    'Product & UX': ['ux ui design', 'usability', 'figma'],
    'CAD & Engineering Design': ['autocad', 'microstation', 'computer-aided design', 'solidworks', '3d'],
    'Business Systems (CRM / ERP)': ['salesforce', 'crm', 'erp'],
}
SKILL_TO_AREA = {s: a for a, skills in TAXONOMY.items() for s in skills}

# Spelling variants of the same skill only. Related but distinct skills are never merged:
# SQL vs SQL Server / MySQL / PostgreSQL, Git vs GitHub / GitLab, Angular vs AngularJS, Java vs JavaScript,
# AI vs Machine Learning vs Deep Learning, AWS vs Azure vs GCP, Windows vs Windows Server, CAD vs AutoCAD.
ALIAS = {'excel': 'microsoft excel', 'machine-learning': 'machine learning', 'react': 'react.js', 'node': 'node.js',
         'nodejs': 'node.js', 'vue': 'vue.js', 'html5': 'html', 'cloud computing': 'cloud',
         'microsoft windows': 'windows', 'microsoft 365': 'office 365', 'oracle': 'oracle database'}

# Tags that are too broad to trust on their own (a domain, a product category or a common word).
# The postings have no description text to check them against, so they are corroborated instead:
# an ambiguous tag counts only if the same posting also has a specific skill from the same course area.
AMBIGUOUS = {'cloud', 'network', 'security', 'vulnerability', 'hardware', 'ai', 'orchestration', 'rest', 'firewall',
             'antivirus software', 'lan', 'cisco', 'usability'}

# Standardized display names (normalized_skill). The raw tag is kept in raw_skill for traceability.
DISPLAY = {
    'aws': 'Amazon Web Services (AWS)', 'azure': 'Microsoft Azure', 'gcp': 'Google Cloud Platform',
    'cloud': 'Cloud Computing', 'devops': 'DevOps', 'ci / cd': 'CI/CD', 'kubernetes': 'Kubernetes', 'docker': 'Docker',
    'terraform': 'Terraform', 'jenkins': 'Jenkins', 'ansible': 'Ansible', 'orchestration': 'Orchestration',
    'helm': 'Helm', 'serverless': 'Serverless', 'gitlab': 'GitLab', 'devsecops': 'DevSecOps',
    'microservices': 'Microservices', 'openstack': 'OpenStack', 'grafana': 'Grafana',
    'sql': 'SQL', 'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'microsoft sql server': 'Microsoft SQL Server',
    'oracle database': 'Oracle Database', 'mongodb': 'MongoDB', 'nosql': 'NoSQL', 'apache spark': 'Apache Spark',
    'apache kafka': 'Apache Kafka', 'etl': 'ETL', 'airflow': 'Apache Airflow', 'big data': 'Big Data',
    'bigquery': 'Google BigQuery', 'data management': 'Data Management',
    'microsoft excel': 'Microsoft Excel', 'spreadsheet': 'Spreadsheets', 'power bi': 'Power BI', 'tableau': 'Tableau',
    'a / b testing': 'A/B Testing', 'matlab': 'MATLAB',
    'ai': 'Artificial Intelligence (AI)', 'machine learning': 'Machine Learning', 'deep learning': 'Deep Learning',
    'tensorflow': 'TensorFlow', 'llm': 'Large Language Models (LLM)',
    'python': 'Python', 'java': 'Java', 'c++': 'C++', 'c sharp': 'C#', 'rust': 'Rust', 'php': 'PHP', 'cobol': 'COBOL',
    'coding': 'Programming', 'software engineering': 'Software Engineering', 'debugging': 'Debugging',
    'unit testing': 'Unit Testing', 'code review': 'Code Review', 'git': 'Git', 'github': 'GitHub',
    'bitbucket': 'Bitbucket', 'dotnet': '.NET', 'spring': 'Spring', 'maven': 'Maven',
    'javascript': 'JavaScript', 'typescript': 'TypeScript', 'react.js': 'React', 'html': 'HTML', 'css': 'CSS',
    'angular': 'Angular', 'angularjs': 'AngularJS', 'node.js': 'Node.js', 'vue.js': 'Vue.js', 'asp.net': 'ASP.NET',
    'jquery': 'jQuery', 'laravel': 'Laravel', 'rest': 'REST APIs', 'api': 'APIs', 'graphql': 'GraphQL',
    'json': 'JSON', 'backend': 'Backend Development', 'fullstack': 'Full-Stack Development',
    'security': 'Cybersecurity (general)', 'firewall': 'Firewalls', 'vulnerability': 'Vulnerability Management',
    'iam': 'Identity & Access Management (IAM)', 'oauth': 'OAuth', 'vpn': 'VPN',
    'waf': 'Web Application Firewall (WAF)', 'antivirus software': 'Antivirus Software', 'jwt': 'JWT',
    'network': 'Computer Networking', 'hardware': 'Computer Hardware', 'windows': 'Windows',
    'windows server': 'Windows Server', 'linux': 'Linux', 'unix': 'Unix', 'macos': 'macOS',
    'powershell': 'PowerShell', 'bash': 'Bash', 'shell script': 'Shell Scripting', 'itil': 'ITIL',
    'cisco': 'Cisco Networking', 'tcp / ip': 'TCP/IP', 'lan': 'Local Area Networks (LAN)',
    'active directory': 'Active Directory', 'office 365': 'Microsoft 365', 'vmware': 'VMware', 'hyper-v': 'Hyper-V',
    'servicenow': 'ServiceNow',
    'microsoft office': 'Microsoft Office', 'microsoft powerpoint': 'Microsoft PowerPoint',
    'microsoft word': 'Microsoft Word', 'sharepoint': 'SharePoint', 'google drive': 'Google Drive', 'slack': 'Slack',
    'jira': 'Jira', 'scrum': 'Scrum', 'confluence': 'Confluence',
    'ux ui design': 'UI/UX Design', 'usability': 'Usability', 'figma': 'Figma',
    'autocad': 'AutoCAD', 'microstation': 'MicroStation', 'computer-aided design': 'Computer-Aided Design (CAD)',
    'solidworks': 'SolidWorks', '3d': '3D Modeling',
    'salesforce': 'Salesforce', 'crm': 'CRM', 'erp': 'ERP',
}
AREA_OF = {DISPLAY[ALIAS.get(s, s)]: a for s, a in SKILL_TO_AREA.items()}   # display name -> course area

# Trend is compared only within sources present at both time points, because the source mix differs by batch
TREND_SOURCES = ['linkedin_us', 'greenhouse_us']
TREND_EARLY, TREND_LATE = '2025-09', '2026-03'


def long_skills(df, ambiguous='corroborated'):
    """One row per (job, skill), keeping only skills listed in TAXONOMY.
    raw_skill is the original tag; skill is the standardized name (normalized_skill).
    ambiguous: 'corroborated' keeps an AMBIGUOUS tag only when the posting also has a specific skill
    from the same course area; 'keep' keeps them all; 'drop' removes them all."""
    ex = df.reset_index(names='job_id').explode('skills').dropna(subset=['skills']).reset_index(drop=True)
    ex['raw_skill'] = ex['skills']
    key = ex['skills'].str.lower().str.strip()
    ex['area'] = key.map(SKILL_TO_AREA)
    key = key.replace(ALIAS)
    ex['is_ambiguous'] = key.isin(AMBIGUOUS)
    ex['skill'] = key.map(DISPLAY)
    ex = ex.dropna(subset=['area'])
    ex = ex.drop_duplicates(['job_id', 'skill'])
    if ambiguous == 'drop':
        ex = ex[~ex['is_ambiguous']]
    elif ambiguous == 'corroborated':
        specific = ex.groupby(['job_id', 'area'])['is_ambiguous'].transform(lambda s: (~s).any())
        ex = ex[~ex['is_ambiguous'] | specific]
    return ex


def ambiguous_check(df):
    """For each ambiguous tag: postings carrying it, and how many are corroborated by a specific skill."""
    allrows = long_skills(df, 'keep')
    kept = long_skills(df, 'corroborated')
    a = allrows[allrows['is_ambiguous']].groupby('skill')['job_id'].nunique().rename('postings_with_tag')
    k = kept[kept['is_ambiguous']].groupby('skill')['job_id'].nunique().rename('corroborated')
    out = pd.concat([a, k], axis=1).fillna(0).astype(int)
    out['corroborated_pct'] = (out['corroborated'] / out['postings_with_tag'] * 100).round(0)
    return out.sort_values('postings_with_tag', ascending=False)


def coverage(df):
    """How many tags and jobs the curated skills cover."""
    raw = df['skills'].explode().dropna().str.lower().str.strip()
    kept = raw.isin(SKILL_TO_AREA)
    jobs = long_skills(df)['job_id'].nunique()   # after the ambiguous-tag check
    return {'total_tags': len(raw), 'kept_tag_share': kept.mean(), 'distinct_tags': raw.nunique(),
            'kept_distinct_tags': raw[kept].nunique(), 'jobs_with_kept_skill': jobs, 'total_jobs': len(df)}


def _trend(sub, df, key):
    """Change in share of postings from the early to the late batch, within source (percentage points)."""
    base = df[df['source'].isin(TREND_SOURCES) & df['batch'].isin([TREND_EARLY, TREND_LATE])]
    denom = base.groupby(['source', 'batch']).size()
    w = base['source'].value_counts(normalize=True)   # fixed source weights
    s = sub[sub['source'].isin(TREND_SOURCES) & sub['batch'].isin([TREND_EARLY, TREND_LATE])]
    num = s.groupby([key, 'source', 'batch'])['job_id'].nunique()
    out = {}
    for k in sub[key].unique():
        share = {b: sum(w[src] * num.get((k, src, b), 0) / denom[(src, b)] for src in TREND_SOURCES)
                 for b in (TREND_EARLY, TREND_LATE)}
        out[k] = {'early_share': share[TREND_EARLY], 'late_share': share[TREND_LATE],
                  'trend_n': int(sum(num.get((k, src, b), 0) for src in TREND_SOURCES for b in (TREND_EARLY, TREND_LATE)))}
    t = pd.DataFrame(out).T
    t['trend_pp'] = (t['late_share'] - t['early_share']) * 100
    return t


def _premium(sub, df, key):
    """Salary premium after controlling for seniority and source: Ridge regression of log(annual salary)
    on group indicators, returned as a percentage."""
    from sklearn.linear_model import Ridge
    m = df.dropna(subset=['salary_annual']).reset_index(names='job_id')
    ind = (sub[sub['job_id'].isin(m['job_id'])].assign(v=1.0)
           .pivot_table(index='job_id', columns=key, values='v', aggfunc='max', fill_value=0.0))
    ind = ind.reindex(m['job_id'], fill_value=0.0)
    ctrl = pd.get_dummies(m[['seniority_label', 'source']].fillna('Unknown'), dtype=float)
    ctrl.index = m['job_id']
    X = pd.concat([ind, ctrl], axis=1)
    coef = Ridge(alpha=3.0).fit(X, np.log(m['salary_annual'].values)).coef_[:ind.shape[1]]
    return pd.Series(np.exp(coef) - 1, index=ind.columns) * 100


def signals(df, key='area', min_jobs=1, ambiguous='corroborated'):
    """Market signals for each course area (key='area') or single skill (key='skill')."""
    sub = long_skills(df, ambiguous)
    g = sub.groupby(key)
    out = pd.DataFrame({
        'jobs': g['job_id'].nunique(),
        'share_pct': g['job_id'].nunique() / len(df) * 100,
        'companies': g['company'].nunique(),
        'industries': g['industry'].nunique(),
        'median_salary': g['salary_annual'].median(),
    })
    if key == 'skill':
        out.insert(0, 'area', g['area'].first())
    out = out[out['jobs'] >= min_jobs]
    sub = sub[sub[key].isin(out.index)]
    out['salary_premium_pct'] = _premium(sub, df, key)
    out = out.join(_trend(sub, df, key)[['early_share', 'late_share', 'trend_pp', 'trend_n']])
    return out.sort_values('jobs', ascending=False)


TREND_MIN_N = 20   # minimum postings in the trend comparison; below this the trend is treated as neutral


def opportunity(sig, w_demand=0.45, w_salary=0.30, w_trend=0.10, w_breadth=0.15):
    """Course opportunity score: each of the four signals is turned into a percentile rank (0 to 1),
    then averaged with the given weights and scaled to 0 to 100.
    Groups with a trend sample below TREND_MIN_N get a neutral trend rank of 0.5."""
    enough = sig['trend_n'] >= TREND_MIN_N
    r = pd.DataFrame({
        'demand': sig['jobs'].rank(pct=True),
        'salary': sig['salary_premium_pct'].rank(pct=True),
        'trend': sig['trend_pp'].where(enough).rank(pct=True).fillna(0.5),
        'breadth': sig['companies'].rank(pct=True),
    })
    w = np.array([w_demand, w_salary, w_trend, w_breadth], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(4, 0.25)
    out = sig.copy()
    out['trend_reliable'] = enough
    for c in r.columns:   # keep the components so the score can be explained
        out['rank_' + c] = r[c].values * 100
    out['score'] = (r.values * w).sum(axis=1) * 100
    return out.sort_values('score', ascending=False)


def cooccurrence(df, min_jobs=8, ambiguous='corroborated'):
    """Pairwise skill co-occurrence: postings together and lift (above 1 = together more often than chance)."""
    sub = long_skills(df, ambiguous)
    keep = sub['skill'].value_counts().loc[lambda s: s >= min_jobs].index
    M = (sub[sub['skill'].isin(keep)].assign(v=1)
         .pivot_table(index='job_id', columns='skill', values='v', aggfunc='max', fill_value=0))
    co = M.T @ M
    n = np.diag(co).astype(float)
    lift = co * M.shape[0] / np.outer(n, n)
    return co, lift


def source_mix(df, ambiguous='corroborated'):
    """Postings per course area and job board (rows: area, columns: source)."""
    sub = long_skills(df, ambiguous).drop_duplicates(['job_id', 'area'])
    return sub.pivot_table(index='area', columns='source', values='job_id', aggfunc='nunique', fill_value=0)


def partners(df, skill, ambiguous='corroborated'):
    """Skills posted together with `skill`: postings together, companies, the share of `skill` postings
    that also list the partner, and lift (above 1 = together more often than chance)."""
    sub = long_skills(df, ambiguous)
    n_all = sub['job_id'].nunique()
    jobs = sub.loc[sub['skill'] == skill, 'job_id'].unique()
    base = sub.groupby('skill')['job_id'].nunique()
    g = sub[sub['job_id'].isin(jobs) & (sub['skill'] != skill)].groupby('skill')
    out = pd.DataFrame({'area': g['area'].first(), 'together': g['job_id'].nunique(), 'companies': g['company'].nunique()})
    out['share_pct'] = out['together'] / len(jobs) * 100
    out['lift'] = (out['together'] / len(jobs)) / (base.reindex(out.index) / n_all)
    return out.sort_values('together', ascending=False), len(jobs)
