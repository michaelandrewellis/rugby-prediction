from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import time
from selenium import webdriver

input_file = 'premiership_data_test.csv'
# set url for Aviva Premiership data
url = "http://www.premiershiprugby.com/aviva-premiership-rugby-results/"

# try selenium
browser = webdriver.Chrome()
browser.get(url)
time.sleep(7)
script = browser.page_source
soup = BeautifulSoup(script,'html.parser')
browser.quit()