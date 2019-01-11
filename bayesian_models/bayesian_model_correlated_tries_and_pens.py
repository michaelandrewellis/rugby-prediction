import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

df = pd.read_csv('expected_points/match_results.csv', index_col=0)
df['date'] = pd.to_datetime(df['date'])
# restrict to 2016-2017 season for now
df = df.loc[(df['date'] > '2016-08-01') & (df['date'] <= '2017-08-01')]
df.head()

df.describe()

teams = df.home_team.unique()
teams = pd.DataFrame(teams, columns=['team'])
teams['i'] = teams.index
teams.head()

df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
df = df.rename(columns={'i': 'i_home'}).drop('team', 1)
df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
df = df.rename(columns={'i': 'i_away'}).drop('team', 1)
df.head()

observed_home_score = df.home_score.values
observed_away_score = df.away_score.values
observed_home_tries_scored = df.home_tries.values
observed_away_tries_scored = df.away_tries.values
observed_home_pens_scored = df.home_pens.values
observed_away_pens_scored = df.away_pens.values
home_team = df.i_home.values
away_team = df.i_away.values
num_teams = len(df.i_home.unique())
num_games = len(home_team)

# initial values
g = df.groupby('i_away')
att_tries_init = np.log(g.away_tries.mean())
g = df.groupby('i_home')
def_tries_init = -np.log(g.away_tries.mean())
g = df.groupby('i_away')
att_pens_init = np.log(g.away_pens.mean())
g = df.groupby('i_home')
def_pens_init = -np.log(g.away_pens.mean())

with pm.Model() as model:
    # global model parameters
    home_tries = pm.Flat('home_tries')  # intercept for home advantage
    home_pens = pm.Flat('home_pens')  # intercept for home advantage
    sd_att = pm.HalfStudentT('sd_att', nu=3, sd=2.5)
    sd_def = pm.HalfStudentT('sd_def', nu=3, sd=2.5)
    sd_att_tries = pm.HalfStudentT('sd_att_tries', nu=3, sd=2.5)
    sd_def_tries = pm.HalfStudentT('sd_def_tries', nu=3, sd=2.5)
    sd_att_pens = pm.HalfStudentT('sd_att_pens', nu=3, sd=2.5)
    sd_def_pens = pm.HalfStudentT('sd_def_pens', nu=3, sd=2.5)
    # sd_att_drops = pm.HalfStudentT('sd_att_drops', nu=3, sd=2.5)
    # sd_def_drops = pm.HalfStudentT('sd_def_drops', nu=3, sd=2.5)

    intercept_tries = pm.Flat('intercept_tries')
    intercept_pens = pm.Flat('intercept_pens')
    # intercept_drops = pm.Flat('intercept_drops')

    # team-specific model parameters
    atts_star = pm.Normal("atts_star", mu=0, sd=sd_att, shape=num_teams)
    defs_star = pm.Normal("defs_star", mu=0, sd=sd_def, shape=num_teams)
    atts_tries_star = pm.Normal("atts_tries_star", mu=0, sd=sd_att_tries, shape=num_teams)
    defs_tries_star = pm.Normal("defs_tries_star", mu=0, sd=sd_def_tries, shape=num_teams)
    atts_pens_star = pm.Normal("atts_pens_star", mu=0, sd=sd_att_pens, shape=num_teams)
    defs_pens_star = pm.Normal("defs_pens_star", mu=0, sd=sd_def_pens, shape=num_teams)
    # atts_drops_star = pm.Normal("atts_drops_star", mu=0, sd=sd_att_drops, shape=num_teams)
    # defs_drops_star = pm.Normal("defs_drops_star", mu=0, sd=sd_def_drops, shape=num_teams)

    atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
    defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
    atts_tries = pm.Deterministic('atts_tries', atts_tries_star - tt.mean(atts_tries_star))
    defs_tries = pm.Deterministic('defs_tries', defs_tries_star - tt.mean(defs_tries_star))
    atts_pens = pm.Deterministic('atts_pens', atts_pens_star - tt.mean(atts_pens_star))
    defs_pens = pm.Deterministic('defs_pens', defs_pens_star - tt.mean(defs_pens_star))
    # atts_drops = pm.Deterministic('atts_drops', atts_drops_star - tt.mean(atts_drops_star))
    # defs_drops = pm.Deterministic('defs_drops', defs_drops_star - tt.mean(defs_drops_star))

    home_theta_tries = tt.exp(intercept_tries + home_tries + atts[home_team] + defs[away_team] + atts_tries[home_team] + defs_tries[away_team])
    away_theta_tries = tt.exp(intercept_tries + atts[away_team] + defs[home_team] + atts_tries[away_team] + defs_tries[home_team])
    home_theta_pens = tt.exp(intercept_pens + home_pens + atts[home_team] + defs[away_team] + atts_pens[home_team] + defs_pens[away_team])
    away_theta_pens = tt.exp(intercept_pens + atts[away_team] + defs[home_team] + atts_pens[away_team] + defs_pens[home_team])

    # likelihood of observed data
    home_tries_scored = pm.Poisson('home_tries_scored', mu=home_theta_tries, observed=observed_home_tries_scored)
    away_tries_scored = pm.Poisson('away_tries_scored', mu=away_theta_tries, observed=observed_away_tries_scored)
    home_pens_scored = pm.Poisson('home_pens_scored', mu=home_theta_pens, observed=observed_home_pens_scored)
    away_pens_scores = pm.Poisson('away_pens_scored', mu=away_theta_pens, observed=observed_away_pens_scored)
    
    
with model:
    trace = pm.sample(1000, tune=1000, cores=3)
    pm.traceplot(trace)

