import pandas as pd
import datetime

all = pd.DataFrame.from_csv('all_events.csv')
all.replace('Gloucester Rugby','Gloucester',inplace=True)
matches = pd.DataFrame.from_csv('../elo/premiership_data_by_team.csv')
matches['date'] = pd.to_datetime(matches['date'])
all['date'] = pd.to_datetime(all['date'])
matches = matches[(matches['date'] > datetime.date(2011,6,24))]
matches = matches[matches['is_copy'] == 0]

home_tries = all[(all['event_type']=='TRY')&(all['scoring_team_location']=='home')].groupby(['home_team','date'])['away_team'].count()
home_tries = home_tries.reset_index()
home_tries.rename(columns={'away_team':'home_tries'},inplace=True)

away_tries = all[(all['event_type']=='TRY')&(all['scoring_team_location']=='away')].groupby(['home_team','date'])['away_team'].count()
away_tries = away_tries.reset_index()
away_tries.rename(columns={'away_team':'away_tries'},inplace=True)

home_pens = all[(all['event_type']=='PEN')&(all['scoring_team_location']=='home')].groupby(['home_team','date'])['away_team'].count()
home_pens = home_pens.reset_index()
home_pens.rename(columns={'away_team':'home_pens'},inplace=True)

away_pens = all[(all['event_type']=='PEN')&(all['scoring_team_location']=='away')].groupby(['home_team','date'])['away_team'].count()
away_pens = away_pens.reset_index()
away_pens.rename(columns={'away_team':'away_pens'},inplace=True)

home_drops = all[(all['event_type']=='DRO')&(all['scoring_team_location']=='home')].groupby(['home_team','date'])['away_team'].count()
home_drops = home_drops.reset_index()
home_drops.rename(columns={'away_team':'home_drops'},inplace=True)

away_drops = all[(all['event_type']=='DRO')&(all['scoring_team_location']=='away')].groupby(['home_team','date'])['away_team'].count()
away_drops = away_drops.reset_index()
away_drops.rename(columns={'away_team':'away_drops'},inplace=True)


for df in [away_tries,away_pens,away_drops]:
    matches = pd.merge(matches,df,how='left',left_on=['team','date'],right_on=['home_team','date'])
    matches.drop('home_team',axis=1,inplace=True)
    
for df in [home_tries,home_pens,home_drops]:
    matches = pd.merge(matches,df,how='left',left_on=['team','date'],right_on=['home_team','date'])
    matches.drop('home_team',axis=1,inplace=True)

matches.fillna(0,inplace=True)
matches.to_csv('match_event_counts.csv')






