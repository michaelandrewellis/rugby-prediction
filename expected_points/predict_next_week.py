import pandas as pd
import datetime
import pickle
from model import prediction

today = datetime.date.today()
next_week = today + datetime.timedelta(days=7)

matches = pd.DataFrame.from_csv('future_matches.csv')
matches['date'] = pd.to_datetime(matches['date'])
next_matches = matches[matches['date'].between(today,next_week)]
while len(next_matches)== 0:
    next_week = next_week + datetime.timedelta(days=7)
    next_matches = matches[matches['date'].between(today, next_week)]

current_ratings = pickle.load(open('ratings_mid_2017_2.pickle','rb'))
predictions = []
for i,match in next_matches.iterrows():
    home_team = match['home_team']
    away_team = match['away_team']
    spread, home_win_prob = prediction(home_team,away_team,n_sims=1000,use_ratings=current_ratings)
    predictions.append([match['date'],home_team,away_team,spread,home_win_prob])
pd.DataFrame(predictions).to_csv('predictions_next_week.csv')


