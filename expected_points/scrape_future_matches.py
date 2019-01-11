import time
import pandas as pd
from selenium import webdriver

url = "http://www.scoresway.com/?sport=rugby&page=round&view=matches&id=3288"

browser = webdriver.Chrome()
browser.get(url)
time.sleep(2)
script = browser.page_source
df = pd.read_html(script)[0]

for i in range(6):
    elem = browser.find_element_by_class_name('next')
    elem.click()
    script = browser.page_source
    df=pd.concat([df,pd.read_html(script)[0]])
    time.sleep(1)
browser.quit()
#df.rename(columns={'Home team':'home_team','Away team':'away_team'},inplace=True)
df.columns = ['day','date','home_team','score/time','away_team','view_events','more_info']
df.drop_duplicates(inplace=True)
df.fillna(method='pad',inplace=True)
df.to_csv('future_matches.csv')