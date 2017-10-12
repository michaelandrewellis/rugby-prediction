import pandas as pd
from math import pow
import scipy.stats as stats
import dateparser

csv_file = 'premiership_data_by_team.csv'
output_file = 'elo.csv'
starting_elo = 1400
promotion_elo = 1400 # 1300
k=5 #25
number_of_iterations = 5 #10
home_bonus = 30 #115

df=pd.DataFrame.from_csv(csv_file)
df.reset_index().drop('index',axis=1).reset_index(inplace=True)
df.rename(index=str,columns={'index':'order'},inplace=True)
teams = set(df['team'].unique())
# deselect repeated matches
df = df[df['is_copy'] == 0]
df['team_elo_i'],df['opp_elo_i'],df['team_elo_n'],df['opp_elo_n'] = 0,0,0,0
elo_dict = {}
for team in teams:
    elo_dict[team] = starting_elo

def expected_score(rating,rating_opp,home_team):
    # add home bonus
    if home_team==True:
        rating = rating + home_bonus
    if home_team==False:
        rating_opp = rating_opp+home_bonus
    return(1/(1+pow(10,(rating_opp-rating)/400)))

def new_rating(rating,rating_opp,result,home_team):
    new_rating = rating + k*(result-expected_score(rating,rating_opp,home_team))
    return(new_rating)

def adjust_elo(team_elo_i, opp_elo_i, team_score, opp_score):
    if team_score>opp_score:
        team_elo_n = new_rating(team_elo_i, opp_elo_i, 1,True)
        opp_elo_n = new_rating(opp_elo_i, team_elo_i, 0,False)
    if team_score<opp_score:
        team_elo_n = new_rating(team_elo_i, opp_elo_i, 0,True)
        opp_elo_n = new_rating(opp_elo_i, team_elo_i, 1,False)
    if team_score==opp_score:
        team_elo_n = new_rating(team_elo_i, opp_elo_i, 0.5,True)
        opp_elo_n = new_rating(opp_elo_i, team_elo_i, 0.5,False)
    return(team_elo_n, opp_elo_n)

def iterate_first_season(df):
    '''
    Warm up model on first season by iterating n times
    '''
    df_first_season = df[df['Season'] == '1998-1999']
    # set elo of teams not in premiership to promotion value
    teams_inc = set(df_first_season['team'].unique())
    for team in elo_dict:
        if team not in teams_inc:
            elo_dict[team] = promotion_elo
    for i in range(0,number_of_iterations):
        for i, row in df_first_season.iterrows():
            team = row['team']
            opp = row['opp']
            team_score = row['team_score']
            opp_score = row['opp_score']
            team_elo_i = elo_dict[team]
            opp_elo_i = elo_dict[opp]
            team_elo_n, opp_elo_n = adjust_elo(team_elo_i, opp_elo_i, team_score, opp_score)
            df.loc[str(i), 'team_elo_i'], df.loc[str(i), 'opp_elo_i'], df.loc[str(i), 'team_elo_n'], df.loc[
                str(i), 'opp_elo_n'] = team_elo_i, opp_elo_i, team_elo_n, opp_elo_n
            elo_dict[team] = team_elo_n
            elo_dict[opp] = opp_elo_n
    
def elo_model(df):
    iterate_first_season(df)
    for i, row in df.iterrows():
        if row['Season'] == '1998-1999':
            continue
        team = row['team']
        opp = row['opp']
        team_score = row['team_score']
        opp_score = row['opp_score']
        team_elo_i = elo_dict[team]
        opp_elo_i = elo_dict[opp]
        team_elo_n, opp_elo_n = adjust_elo(team_elo_i, opp_elo_i, team_score, opp_score)
        df.loc[str(i), 'team_elo_i'], df.loc[str(i), 'opp_elo_i'], df.loc[str(i), 'team_elo_n'], df.loc[
            str(i), 'opp_elo_n'] = team_elo_i, opp_elo_i, team_elo_n, opp_elo_n
        df.loc[str(i), 'prediction'] = expected_score(team_elo_i, opp_elo_i,True)
        elo_dict[team] = team_elo_n
        elo_dict[opp] = opp_elo_n
        
def evaluate_model(df):
    sse=0
    count=0
    for i, row in df.iterrows():
        if row['Season'] == '1998-1999':
            continue
        count += 1
        sse+=abs(row['prediction']-row['result']) ** 2
    mean_sse = sse/count
    print(sse,mean_sse)
        
        
        

elo_model(df)
evaluate_model(df)
df_copy = df.copy()
df_copy.rename(columns={"team_score":"opp_score","opp_score":"team_score","opp_elo_i": "team_elo_i", "opp_elo_n": "team_elo_n","team_elo_i": "opp_elo_i","team_elo_n": "opp_elo_n","team":"opp","opp":"team"})
df_copy['is_copy'] = 1
df = pd.concat([df,df_copy])
df.to_csv(output_file)