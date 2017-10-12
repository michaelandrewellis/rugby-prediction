from bs4 import BeautifulSoup
import requests
import pandas as pd

# set url for Aviva Premiership data
url = "http://www.premiershiprugby.com/aviva-premiership-rugby-results/"

# get soup
page = requests.get(url)
soup = BeautifulSoup(page.text,'html.parser')

# create dictionary mapping year to its corresponding code
script = page.text
data = script.split('dropdowndata85837498[0]')[2:-1]
codes = [x.split('=')[0].split('\'')[1] for x in data]
years = [x.split('=')[1][2:-4] for x in data]
year_code_dict = {}
for year,code in zip(years,codes):
    year_code_dict[year] = code
    
for year in years.__reversed__():
    code = year_code_dict[year]
    table = soup.find('table', class_='fl85837498 fl85837498-'+code+' sortable')
    df_new = pd.read_html(str(table), skiprows=1)[0]
    df_new['Season'] = year
    if 'df' in locals():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        df = pd.concat([df,df_new])
    else:
        df = df_new

df = pd.concat([df.ix[:,0:6],df['Season']],axis=1)
df.columns = ['Data','Time','Home','Score','Away','Venue','Season']
df.dropna(thresh=4,inplace=True)
df.to_csv('premiership_data.csv')
