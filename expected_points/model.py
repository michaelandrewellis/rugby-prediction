#!/usr/bin/env python3

import pandas as pd
import datetime
import scipy.stats as stats
import statistics
import numpy as np
import pickle
from random import choice
import sys,os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

df = pd.DataFrame.from_csv(os.path.join(__location__, 'match_results.csv'))
df['spread_prediction'] = np.nan
df['win_prob'] = np.nan
df['home_scores'] = df['home_tries']+df['home_pens']
df['away_scores'] = df['away_tries']+ df['away_pens']

a = 0.93 # for exponential smoothing
n_simulations = 100
conversion_rate = 0.71
home_boost = 1.3 # 1.277
mean_home_tries = 2.35
mean_home_pens = 2.35
mean_away_tries = 1.8
mean_away_pens = 2.05


mean_pens = 2.2
mean_tries = 2.07

#for single score
mean_home_scores = 4.7
mean_away_scores = 3.85
mean_scores = 4.3

starting_ratings = {'TRY_OFF':1.5,'TRY_DEF':3,'PEN_OFF':1.5,'PEN_DEF':3}


teams = set(df['home_team'].unique())

# set initial ratings
ratings = {}
for team in teams:
    ratings[team] = starting_ratings.copy()
    
# split data
initialising_data = df[pd.to_datetime(df['date'])<datetime.date(2012,6,24)].copy()
test_data = df[pd.to_datetime(df['date']).between(datetime.date(2012,6,24),datetime.date(2017,6,24))].copy()


home_tries_predictions = []

def simulate(n,exp_home_tries,exp_away_tries,exp_home_pens,exp_away_pens):
    differences = []
    for i in range(n):
        home_tries = stats.poisson.rvs(exp_home_tries)
        away_tries = stats.poisson.rvs(exp_away_tries)
        home_conversions = stats.binom.rvs(home_tries,conversion_rate)
        away_conversions = stats.binom.rvs(away_tries,conversion_rate)
        home_pens = stats.poisson.rvs(exp_home_pens)
        away_pens = stats.poisson.rvs(exp_away_pens)
        home_score = 5*home_tries+2*home_conversions+3*home_pens
        away_score = 5*away_tries+2*away_conversions+3*away_pens
        difference = home_score - away_score
        differences.append(difference)
    spread = statistics.median(differences)
    home_win_prob = sum(difference > 0  for difference in differences)/len(differences)
    return(spread,home_win_prob,differences)

def exp_score_rates(home_team,away_team,ratings):
    exp_home_tries = ratings[home_team]['TRY_OFF']/mean_tries*ratings[away_team]['TRY_DEF']/mean_tries*mean_home_tries
    exp_away_tries = ratings[away_team]['TRY_OFF']/mean_tries * ratings[home_team]['TRY_DEF'] / mean_tries*mean_away_tries
    exp_home_pens = ratings[home_team]['PEN_OFF'] / mean_pens * ratings[away_team][
        'PEN_DEF'] / mean_pens * mean_home_pens
    exp_away_pens = ratings[away_team]['PEN_OFF'] / mean_tries * ratings[home_team][
        'PEN_DEF'] / mean_pens * mean_away_pens
    return(exp_home_tries,exp_away_tries,exp_home_pens,exp_away_pens)

def prediction(home_team,away_team,n_sims = n_simulations,use_ratings=None,inc_spread_data=False):
    if use_ratings == None:
        global ratings
    else:
        ratings = use_ratings
    exp_home_tries, exp_away_tries, exp_home_pens, exp_away_pens = exp_score_rates(home_team,away_team,ratings)
    spread,home_win_prob,spread_data = simulate(n_sims,exp_home_tries,exp_away_tries,exp_home_pens,exp_away_pens)
    if inc_spread_data:
        return(spread,home_win_prob,spread_data)
    else:
        return(spread,home_win_prob)

def simulate_single(n,exp_home_scores,exp_away_scores):
    differences = []
    for i in range(n):
        home_scores = stats.poisson.rvs(exp_home_scores)
        away_scores = stats.poisson.rvs(exp_away_scores)
        home_tries = stats.binom.rvs(home_scores,0.5)
        away_tries = stats.binom.rvs(away_scores,0.5)
        home_conversions = stats.binom.rvs(home_tries, conversion_rate)
        away_conversions = stats.binom.rvs(away_tries, conversion_rate)
        home_pens = home_scores-home_tries
        away_pens = away_scores-away_tries
        home_score = 5*home_tries+2*home_conversions+3*home_pens
        away_score = 5*away_tries+2*away_conversions+3*away_pens
        difference = home_score - away_score
        differences.append(difference)    
    spread = statistics.median(differences)
    home_win_prob = sum(difference >= 0  for difference in differences)
    return(spread,home_win_prob)

def prediction_single_score(home_team,away_team):
    exp_home_scores = ratings[home_team]['OFF'] / mean_scores * ratings[away_team][
        'DEF'] / mean_scores * mean_home_scores
    exp_away_scores = ratings[away_team]['OFF'] / mean_scores * ratings[home_team][
        'DEF'] / mean_scores * mean_away_scores
    spread,home_win_prob = simulate_single(n_simulations,exp_home_scores,exp_away_scores)
    return(spread,home_win_prob)

def update_ratings_single(home_team,away_team,home_scores,away_scores):
    global ratings
    new_ratings = ratings
    print(ratings[home_team]['OFF'],ratings[away_team]['OFF'])

    print(home_team, a * ratings[home_team]['OFF'],(1 - a) * (mean_scores * home_scores / (
                                             ratings[away_team]['DEF'] / mean_scores * mean_home_scores)),a * ratings[home_team]['OFF'] + \
                                             (1 - a) * (mean_scores * home_scores / (
                                             ratings[away_team]['DEF'] / mean_scores * mean_home_scores)))
    print(away_team,a * ratings[away_team]['OFF'],(1 - a) * (
                                                 mean_scores * away_scores / (
                                                 ratings[away_team]['OFF'] / mean_scores * mean_away_scores)), a * ratings[away_team]['OFF'] + \
                                             (1 - a) * (
                                                 mean_scores * away_scores / (
                                                 ratings[away_team]['OFF'] / mean_scores * mean_away_scores)))
    new_ratings[home_team]['OFF'] = a * ratings[home_team]['OFF'] + \
                                             (1 - a) * (mean_scores * home_scores / (
                                             ratings[away_team]['DEF'] / mean_scores * mean_home_scores))
    new_ratings[away_team]['DEF'] = a * ratings[away_team]['DEF'] + \
                                             (1 - a) * (
                                                 mean_scores * home_scores / (
                                                 ratings[home_team]['OFF'] / mean_scores * mean_home_scores))
    new_ratings[away_team]['OFF'] = a * ratings[away_team]['OFF'] + \
                                             (1 - a) * (
                                                 mean_scores * away_scores / (
                                                 ratings[home_team]['DEF'] / mean_scores * mean_away_scores))
    new_ratings[home_team]['DEF'] = a * ratings[home_team]['DEF'] + \
                                             (1 - a) * (
                                                 mean_scores * away_scores / (
                                                 ratings[away_team]['OFF'] / mean_scores * mean_away_scores))
    print(new_ratings[home_team],new_ratings[away_team])
    ratings = new_ratings
    
def update_ratings(home_team,away_team,home_tries,away_tries,home_pens,away_pens, ratings):
    new_ratings = ratings
    for event,home,away,mean,mean_home,mean_away in zip(['TRY','PEN'],[home_tries,home_pens],[away_tries,away_pens],[mean_tries,mean_pens],[mean_home_tries,mean_home_pens],[mean_away_tries,mean_away_pens]):
        '''home_AGS = ((home - ratings[away_team][event+'_DEF']) / (max(0.25,ratings[away_team][event+'_DEF'] * 0.4 + 0.5)))*(mean*0.4+0.5)+mean
        home_AGA = ((away - ratings[away_team][event+'_OFF']) / (max(0.25,ratings[away_team][event+'_OFF'] * 0.4 + 0.5)))*(mean*0.4+0.5)+mean
        away_AGS = ((away - ratings[home_team][event+'_DEF']) / (max(0.25,ratings[home_team][event+'_DEF'] * 0.4 + 0.5)))*(mean*0.4+0.5)+mean
        away_AGA = ((home - ratings[home_team][event+'_OFF']) / (max(0.25,ratings[home_team][event+'_OFF'] * 0.4 + 0.5)))*(mean*0.4+0.5)+mean
        new_ratings[home_team][event+'_OFF'] = a * ratings[home_team][event+'_OFF'] +\
                                        (1-a)*home_AGS
        new_ratings[home_team][event + '_DEF'] = a * ratings[home_team][event + '_DEF'] + \
                                                 (1 - a) * home_AGA
        new_ratings[away_team][event + '_OFF'] = a * ratings[away_team][event + '_OFF'] + \
                                                 (1 - a) * away_AGS                                        
        new_ratings[away_team][event+'_DEF'] = a * ratings[away_team][event+'_DEF'] + \
                                        (1 - a) * away_AGA'''

        new_ratings[home_team][event+'_'+'OFF'] = a * ratings[home_team][event+'_'+'OFF'] + \
                                        (1 - a) * (mean * home / (
                                            ratings[away_team][event+'_'+'DEF'] / mean * mean_home))
        new_ratings[away_team][event+'_'+'DEF'] = a * ratings[away_team][event+'_'+'DEF'] + \
                                        (1 - a) * (
                                            mean * home / (
                                                ratings[home_team][event+'_'+'OFF'] / mean * mean_home))
        new_ratings[away_team][event+'_'+'OFF'] = a * ratings[away_team][event+'_'+'OFF'] + \
                                        (1 - a) * (
                                            mean * away / (
                                                ratings[home_team][event+'_'+'DEF'] / mean * mean_away))
        new_ratings[home_team][event+'_'+'DEF'] = a * ratings[home_team][event+'_'+'DEF'] + \
                                        (1 - a) * (
                                            mean * away / (
                                                ratings[away_team][event+'_'+'OFF'] / mean * mean_away))
    ratings = new_ratings
    return(ratings)
    
def initialise(n_iter,df_init,ratings):
    for j in range(0,n_iter):
        for i,row in df_init.iterrows():
            spread,home_win_prob = prediction(row['home_team'],row['away_team'])
            df_init.loc[i,'spread_prediction'] = spread
            df_init.loc[i,'win_prob'] = home_win_prob
            ratings = update_ratings(row['home_team'],row['away_team'],row['home_tries'],row['away_tries'],row['home_pens'],row['away_pens'],ratings)
    df_init['difference'] = df_init['spread_prediction']-(df_init['home_score']-df_init['away_score'])
    return(ratings)
    
def test(df_test,ratings):
    for i,row in df_test.iterrows():
        spread, home_win_prob = prediction(row['home_team'], row['away_team'])
        df_test.loc[i, 'spread_prediction'] = spread
        df_test.loc[i, 'win_prob'] = home_win_prob
        ratings = update_ratings(row['home_team'], row['away_team'], row['home_tries'], row['away_tries'], row['home_pens'],
                       row['away_pens'],ratings)
    df_test['error'] = df_test['spread_prediction'] - (df_test['home_score'] - df_test['away_score'])
    return(ratings)
    
##### SIMULATE SEASON ######    
    
def simulate_once(exp_home_tries,exp_away_tries,exp_home_pens,exp_away_pens):
    '''Simulate match only once. For use in simulating whole season.'''
    home_tries = stats.poisson.rvs(exp_home_tries)
    away_tries = stats.poisson.rvs(exp_away_tries)
    home_conversions = stats.binom.rvs(home_tries, conversion_rate)
    away_conversions = stats.binom.rvs(away_tries, conversion_rate)
    home_pens = stats.poisson.rvs(exp_home_pens)
    away_pens = stats.poisson.rvs(exp_away_pens)
    home_score = 5 * home_tries + 2 * home_conversions + 3 * home_pens
    away_score = 5 * away_tries + 2 * away_conversions + 3 * away_pens
    return(home_score,away_score,home_tries,away_tries,home_conversions,away_conversions,home_pens,away_pens)

def current_table():
    ratings_file = os.path.join(__location__, 'ratings2017.pickle')
    matches_file = os.path.join(__location__, 'match_results.csv')
    ratings = pickle.load(open(ratings_file, 'rb'))
    matches = pd.DataFrame.from_csv(matches_file)
    matches['date']=pd.to_datetime(matches['date'])
    matches = matches[matches['date']>datetime.date(2017,6,24)]
    teams = matches['home_team'].unique()
    points = {}
    for team in teams:
        points[team] = 0
    for i,row in matches.iterrows():
        home_team,away_team,home_score,away_score = row['home_team'], row['away_team'],row['home_score'],row['away_score']
        home_tries,away_tries,home_pens,away_pens = row['home_tries'],row['away_tries'],row['home_pens'],row['away_pens']
        add_match_points(points, home_team, away_team, home_score, away_score, home_tries, away_tries)   
        ratings = update_ratings(home_team, away_team, home_tries, away_tries, home_pens, away_pens,ratings)
    with open(os.path.join(__location__,'ratings_mid_2017.pickle'), 'wb') as f:
        pickle.dump(ratings, f)
    return(points)

def add_match_points(points,home_team,away_team,home_score,away_score,home_tries,away_tries):
    if home_score > away_score:
        points[home_team] += 4
        if (home_score - away_score) <= 7:
            points[away_team] += 1
    elif away_score > home_score:
        points[away_team] += 4
        if (away_score - home_score) <= 7:
            points[home_team] += 1
    elif away_score == home_score:
        points[home_team] += 2
        points[away_team] += 2
    if home_tries >= 4:
        points[home_team] += 1
    if away_tries >= 4:
        points[away_team] += 1
    return(points)

def knockout_winner(home_team,away_team,ratings):
    exp_home_tries, exp_away_tries, exp_home_pens, exp_away_pens = exp_score_rates(home_team, away_team,ratings)
    home_score, away_score, home_tries, away_tries, home_conversions, away_conversions, home_pens, away_pens = \
        simulate_once(exp_home_tries, exp_away_tries, exp_home_pens, exp_away_pens)
    #ratings = update_ratings(home_team, away_team, home_tries, away_tries, home_pens, away_pens,ratings)
    if home_score>away_score:
        return(home_team,ratings)
    elif away_score>home_score:
        return(away_team,ratings)
    elif away_score==home_score:
        return(choice([home_team,away_team]),ratings)
    
def simulate_remaining_season(n_iterations,points_table,ratings_file,updating=False):
    future_matches_file = os.path.join(__location__, 'future_matches.csv')
    rankings = {}
    playoff_counts = {}
    winners_counts = {}
    for key in points_table:
        rankings[key] = []
        playoff_counts[key] = 0
        winners_counts[key] = 0
    matches = pd.DataFrame.from_csv(future_matches_file)
    matches = matches[matches['score/time'].str.contains(':')] # Select only matches that are yet to happen and so have time instead of score
    for k in range(n_iterations):
        ratings = pickle.load(open(ratings_file, 'rb'))
        new_points_table = points_table.copy()
        for i,row in matches.iterrows():
            home_team,away_team = row['home_team'],row['away_team']
            exp_home_tries, exp_away_tries, exp_home_pens, exp_away_pens = exp_score_rates(home_team, away_team,ratings)
            home_score, away_score, home_tries, away_tries, home_conversions, away_conversions, home_pens, away_pens =\
                simulate_once(exp_home_tries, exp_away_tries, exp_home_pens, exp_away_pens)
            if updating == True:
                ratings = update_ratings(home_team, away_team, home_tries, away_tries, home_pens, away_pens, ratings)
            else:
                pass
            new_points_table = add_match_points(new_points_table,home_team,away_team,home_score,away_score,home_tries,away_tries)
        for rank, key in enumerate(sorted(new_points_table, key=new_points_table.get, reverse=True), 1):
            rankings[key].append(rank)
            if rank == 4:
                fourth = key
            elif rank ==3:
                third = key
            elif rank == 2:
                second = key
            elif rank == 1:
                first = key
            if rank <= 4:
                playoff_counts[key] += 1
        finalist1,ratings = knockout_winner(first,fourth,ratings)
        finalist2, ratings = knockout_winner(second,third,ratings)
        final_host = choice([finalist1,finalist2])
        final_guest = list(set([finalist1,finalist2]).difference(set(final_host)))[0]
        winner, ratings = knockout_winner(final_host,final_guest,ratings)
        winners_counts[winner] += 1
    return(rankings,playoff_counts,winners_counts)

def predict_remaining_season(ratings_file,output_file,n_iterations,updating=False):
    points_table = current_table()
    iterations = n_iterations
    rankings, playoff_counts, winner_counts = simulate_remaining_season(iterations, points_table,ratings_file,updating=updating)
    df = pd.DataFrame()
    df_rankings = pd.DataFrame.from_dict(rankings)
    df_rankings.to_csv(os.path.join(__location__, 'rankings.csv'))
    for key in rankings:
        df = df.append([[key, playoff_counts[key] / iterations, winner_counts[key] / iterations]])
    df.columns = ['Team','Top 4 Finish','Grand Final Winner']
    df[['Top 4 Finish','Grand Final Winner']] = df[['Top 4 Finish','Grand Final Winner']].round(decimals=4)*100
    df.to_csv(output_file, index=False)

if __name__ == "__main__":  
    '''ratings = initialise(10,initialising_data,ratings)
    ratings = test(test_data, ratings)
    with open('ratings2017.pickle','wb') as f:
        pickle.dump(ratings,f)'''
    predict_remaining_season(os.path.join(__location__, 'ratings_mid_2017.pickle'),os.path.join(__location__,'season_probs_updating_ratings.csv'),10000,updating=True)
    
    

        
        


