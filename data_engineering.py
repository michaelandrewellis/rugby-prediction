import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import dateparser

df = pd.DataFrame.from_csv('premiership_data.csv')
df['Home Score'] = df['Score'].apply(lambda x: int(x.split('-')[0]))
df['Away Score'] = df['Score'].apply(lambda x: int(x.split('-')[1]))
df.drop('Score',axis=1,inplace=True)
df = df.reset_index().drop('index',axis=1).reset_index()
df.rename(index=str,columns={'index':'match_id'},inplace=True)

def club_renaming(club):
    renaming_dict = {}
    renaming_dict['The Sharks'] = 'Sale Sharks'
    renaming_dict['London Wasps'] = 'Wasps'
    renaming_dict['Gloucester Rugby'] = 'Gloucester'
    renaming_dict['Bristol Shoguns'] = 'Bristol Rugby'
    renaming_dict['Leeds Tykes'] = 'Leeds Carnegie'
    renaming_dict['NEC Harlequins'] = 'Harlequins'
    if club in renaming_dict:
        club = renaming_dict[club]
    return(club)
    
    

# function for results, not used yet
def result_func(team_score, opp_score):
    if team_score>opp_score:
        return(1)
    if team_score<opp_score:
        return(0)
    if team_score==opp_score:
        return(0.5)

def new_date(date,season):
    old_date=dateparser.parse(date.split(' ')[2])
    if old_date.month < 8:
        year = season.split('-')[1]
    if old_date.month >= 8:
        year = season.split('-')[0]
    new_date = date+' '+year
    return(new_date)

# Rename clubs
df['Home'] = df['Home'].apply(club_renaming)
df['Away'] = df['Away'].apply(club_renaming)
df2 = df.copy()
df[['team', 'opp','team_score','opp_score']] = df[['Home','Away','Home Score','Away Score']]
df2[['team', 'opp','team_score','opp_score']] = df[['Away','Home','Away Score','Home Score']]
df2['is_copy'] = 1
df2['location'] = 'A'
df['is_copy'] = 0
df['location'] = 'H'
df = pd.concat([df, df2],ignore_index=True)
df['result'] = df[['team_score','opp_score']].apply(lambda x: result_func(x[0],x[1]),axis=1)

df['Date'] = df[['Data','Season']].apply(lambda x: new_date(x[0],x[1]),axis=1)
df['Date'] = df['Date'].apply(lambda x: dateparser.parse(x))
df.drop(['Data','Home','Away','Home Score','Away Score'],axis=1,inplace=True)


    
df.to_csv('premiership_data_by_team.csv')
