#!/usr/bin/env python3

import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from merging import events_to_results
import re

url = "http://www.scoresway.com/?sport=rugby&page=round&view=matches&id=3288"


old_results_df = pd.DataFrame.from_csv('match_results.csv')
old_results_df['date'] = pd.to_datetime(old_results_df['date'])
    
def get_events(url):
    browser = webdriver.Chrome()
    browser.get(url)
    time.sleep(2)
    script = browser.page_source
    #df = pd.read_html(script)[0]
    soup = BeautifulSoup(script, 'html.parser')
    nodes = soup.find_all('td', 'info-button button')
    for node in nodes:
        match_url = "http://www.scoresway.com/" + node.find('a')['href']
        text = requests.get(match_url).text
        soup = BeautifulSoup(text, 'html.parser')
        teams = soup.findAll('h3', 'thick')
        home = teams[0].text.strip()
        away = teams[2].text.strip()
        date = soup.find('div', 'details clearfix').findAll('dd')[1].text
        event_table = pd.read_html(text)[0]
        try:
            event_table.columns = ['home_event', 'score', 'away_event']
            event_table['home_team'] = home
            event_table['away_team'] = away
            event_table['date'] = date
            event_table['half'] = 'none'
            event_table.loc[1:event_table[event_table['home_event'] == '2nd Half'].index[0], 'half'] = 'first'
            event_table.loc[event_table[event_table['home_event'] == '2nd Half'].index[0] + 1:, 'half'] = 'second'
            event_table.drop([0, event_table[event_table['home_event'] == '2nd Half'].index[0]], inplace=True)
            if (event_table['home_event'] == 'Extra-time').any():
                event_table.drop(event_table[event_table['home_event'] == 'Extra-time'].index[0], inplace=True)
            if (event_table['home_event'] == 'None').any():
                event_table.drop(event_table[event_table['home_event'] == 'None'].index[0], inplace=True)
            event_table.reset_index(drop=True, inplace=True)
            event_table['home_score'] = event_table['score'].apply(lambda x: float(x.split()[0]))
            event_table['away_score'] = event_table['score'].apply(lambda x: float(x.split()[2]))
            event_table['scoring_team_location'] = event_table[['home_event', 'away_event']].fillna('').apply(
                lambda x: 'home' if x[0] is not '' else 'away',
                axis=1)
            event_table['scoring_team'] = event_table[['home_event', 'away_event']].fillna('').apply(
                lambda x: home if x[0] is not '' else away,
                axis=1)
            event_table['event'] = event_table[['home_event', 'away_event']].fillna('').apply(lambda x: x[0] + x[1],
                                                                                              axis=1)
            event_table['time'] = event_table['event'].apply(lambda x: re.search(r'(\d+)\'', x).group(1))
            # Change this to TRY and CONV instead of PT
            event_table['event_type'] = event_table['event'].apply(
                lambda x: re.search(r'\((\w\w\w)\)', x).group(1) if re.search(r'\((\w\w\w)\)', x) else 'PT')
            event_table['conversion'] = event_table['event'].apply(lambda x: 'CONV' if 'Conv' in x else '')
            event_table['date'] = pd.to_datetime(event_table['date'])
        except:
            print(event_table)
        try:
            df = pd.concat([df, event_table])
        except:
            df = event_table
    browser.quit()
    return(df)

df = get_events(url)
recent_results = events_to_results(df)
all_results = pd.concat([old_results_df,recent_results]).drop_duplicates()
all_results.to_csv('match_results.csv')