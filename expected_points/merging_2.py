import pandas as pd
import glob

for file in glob.glob('events_*.csv'):
    new_df = pd.DataFrame.from_csv(file).drop_duplicates()
    try:
        df = pd.concat([df,new_df])
    except:
        df = new_df
all = df.iloc[:,6:-2].dropna(thresh=5,axis=0)
all.to_csv('all_events_2.csv')
matches = all[['home_team','date','away_team','home_score','away_score']].drop_duplicates(['home_team','date','away_team'],keep='last')

home_tries = all[(all['event_type']=='TRY')&(all['scoring_team_location']=='home')].groupby(['home_team','date','away_team'])['half'].count()
home_tries = home_tries.reset_index()
home_tries.rename(columns={'half':'home_tries'},inplace=True)

away_tries = all[(all['event_type']=='TRY')&(all['scoring_team_location']=='away')].groupby(['home_team','date','away_team'])['half'].count()
away_tries = away_tries.reset_index()
away_tries.rename(columns={'half':'away_tries'},inplace=True)

home_pens = all[(all['event_type']=='PEN')&(all['scoring_team_location']=='home')].groupby(['home_team','date','away_team'])['half'].count()
home_pens = home_pens.reset_index()
home_pens.rename(columns={'half':'home_pens'},inplace=True)

away_pens = all[(all['event_type']=='PEN')&(all['scoring_team_location']=='away')].groupby(['home_team','date','away_team'])['half'].count()
away_pens = away_pens.reset_index()
away_pens.rename(columns={'half':'away_pens'},inplace=True)

for df in [away_tries, away_pens]:
    matches = pd.merge(matches, df, how='left', on=['home_team', 'date','away_team'])

for df in [home_tries, home_pens]:
    matches = pd.merge(matches, df, how='left', on=['home_team', 'date','away_team'])

matches.fillna(0, inplace=True)
matches['date'] = pd.to_datetime(matches['date'])
matches.sort_values('date').to_csv('match_results.csv')