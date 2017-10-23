import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import sys

input_file = 'elo.csv'
output_file = 'elo_table_by_week.csv'
interpolated_output_file = 'elo_table_by_week_interpolated.csv'

df = pd.DataFrame.from_csv(os.path.join(sys.path[0],input_file))

def next_weekday(d, weekday):
    days_ahead = (weekday - d.weekday())%7 
    return d + datetime.timedelta(days_ahead)

df.Date = pd.to_datetime(df.Date)
df.Date = df.Date.apply(lambda x: next_weekday(x,3))
table = df.pivot_table(index='Date',columns='team',values='team_elo_n')

# Rename for d3 to handle more nicely
table.index.rename('date',inplace=True)
table.to_csv(os.path.join(sys.path[0],output_file))
table.interpolate(limit = 10).to_csv(os.path.join(sys.path[0],interpolated_output_file))
