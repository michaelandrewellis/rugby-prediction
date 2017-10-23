import pandas as pd
import datetime
import pickle
from model import prediction

today = datetime.date.today()
next_week = today + datetime.timedelta(days=7)

matches = pd.DataFrame.from_csv('future_matches.csv')
matches['Date'] = pd.to_datetime(matches['Date'])
matches = matches[matches['Date'].between(today,next_week)]
current_ratings = pickle.load(open('ratings_mid_2017.pickle','rb'))
for i,match in matches.iterrows():
    home_team = match['home_team']
    away_team = match['away_team']
    spread, home_win_prob = prediction(home_team,away_team,n_sims=1000,use_ratings=current_ratings)
    print(home_team,away_team,spread,home_win_prob)


