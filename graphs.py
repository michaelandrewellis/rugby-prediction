import pandas as pd
import matplotlib.pyplot as plt
import datetime

df = pd.DataFrame.from_csv('elo.csv')

def next_weekday(d, weekday):
    days_ahead = (weekday - d.weekday())%7 
    return d + datetime.timedelta(days_ahead)

df.Date = pd.to_datetime(df.Date)
df.Date = df.Date.apply(lambda x: next_weekday(x,3))
table = df.pivot_table(index='Date',columns='team',values='team_elo_n')
# Rename for d3 to handle more nicely
df.index.rename('date',inplace=True)
table.to_csv('elo_table_by_week.csv')
table.interpolate(limit = 10).to_csv('elo_table_by_week_interpolated_10.csv')
