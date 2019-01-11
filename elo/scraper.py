from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import sys
import re
import time
from selenium import webdriver


output_file = 'premiership_data_test.csv'
# set url for Aviva Premiership data
url = "http://www.premiershiprugby.com/aviva-premiership-rugby-results/"

# try selenium
browser = webdriver.Chrome()
browser.get(url)
time.sleep(7)
script = browser.page_source
soup = BeautifulSoup(script,'html.parser')
browser.quit()

# create dictionary mapping year to its corresponding code
#script = page.text
#data = script.split('dropdowndata85837498[0]')[2:-1]
data = script.split('dropdowndata')[3:-3] #\d+[0]
codes = [x.split('=')[0].split('\'')[1] for x in data]
years = [x.split('=')[1][2:-3] for x in data]
year_code_dict = {}
for year,code in zip(years,codes):
    year_code_dict[year] = code
    

prefix = soup.findAll('select')[0]['id'][0:-2]
  
for year in years.__reversed__():
    code = year_code_dict[year]
    table = soup.findAll('table', {"class":prefix + ' ' + prefix +'-'+ code + ' sortable'})[0]
    df_new = pd.read_html(str(table), skiprows=1)[0].ix[:,0:6]
    df_new.columns = ['Data', 'Time', 'Home', 'Score', 'Away', 'Venue']
    df_new['Season'] = year
    df_new.dropna(thresh=4, inplace=True)
    if 'df' in locals():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        df = pd.concat([df,df_new])
    else:
        df = df_new
        
#df = pd.concat([df.ix[:,0:6],df['Season']],axis=1)
#df.columns = ['Data','Time','Home','Score','Away','Venue','Season']
#df.dropna(thresh=4,inplace=True)
df.to_csv(os.path.join(sys.path[0],output_file))
