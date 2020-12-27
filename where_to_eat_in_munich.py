# You can find the blog post about this code at: www.gabriel-berardi.com/post/where-to-eat-in-munich

# Import required ibraries

import pandas as pd
import requests
import time
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
import geopandas as gpd
from math import e
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from random import randint

# Set the base url and the first page to scrape

base_url = 'URL'
first_page = 'Extension'

# This code block retrieves the url extensions for all restaurants

next_page = urljoin(base_url, first_page)
page_exts = []
i = 0

loop = True
while loop == True:
    i += 1
    print(f'Now scraping page number {i}...')
    time.sleep(randint(10,15))
    r = requests.get(next_page)
    soup = BeautifulSoup(r.text, "html.parser")

    for url in soup.find_all(class_ = "_15_ydu6b"):
        page_exts.append(url['href'])
        
    try:
        next_button = soup.find(class_ = 'nav next rndBtn ui_button primary taLnk')['href']
        next_page = urljoin(base_url, next_button)
    except TypeError:
        print('Last Page Reached...')
        loop = False

# This code block extracts the name, location, rating, number of reviews and price range
# from all the restaurants

rest_name = []
rest_loc = []
rest_rating = []
rest_norat = []
rest_price = []

for page_ext in page_exts:
    print(f'Now scraping restaurant number {page_exts.index(page_ext)}...')
    time.sleep(randint(1,3))
    r = requests.get(urljoin(base_url, page_ext))
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        rest_name.append(soup.find(class_ = '_3a1XQ88S').text)
    except AttributeError:
        rest_name.append(None)
    try:
        rest_loc.append(soup.find(class_ = '_2saB_OSe').text)
    except AttributeError:
        rest_loc.append(None)
    try:
        rest_rating.append(soup.find(class_ = 'r2Cf69qf').text)
    except AttributeError:
        rest_rating.append(None)
    try:
        rest_norat.append(soup.find(class_ = '_10Iv7dOs').text)
    except AttributeError:
        rest_norat.append(None)
    try:
        rest_price.append(soup.find(class_ = '_2mn01bsa').text)
    except AttributeError:
        rest_price.append(None)

# Create the dataframe from the scraped data and save it to a csv file

df = pd.DataFrame(data = list(zip(rest_name,
                                  rest_loc,
                                  rest_rating,
                                  rest_norat,
                                  rest_price)),
                  columns = ['name',
                             'address',
                             'rating',
                             'number_reviews',
                             'price'])
df.to_csv('restaurants.csv')

# Reading in the raw scraped data

df = pd.read_csv('munich restaurants.csv', index_col = 'Unnamed: 0')

# Delete all duplicate rows and only keep the first entry

df = df.drop_duplicates(keep = 'first')

# Checking for null values

print(df.info())

# Drop all rows contain null values

df = df.dropna()

# Delete all rows where the 'Price Range' information is wrong

df = df[df['price'].str.contains('$', regex = False)]

# Format the 'No. of Ratings' column

df.loc[:, 'number_reviews'] = df['number_reviews'].apply(lambda x: re.sub('[^0-9]','', x))
df['number_reviews'] = df['number_reviews'].astype(int)

# Format the 'Rating' column

df['rating'] = df['rating'].astype(float)

# Format the 'Price Range' column

df['price'] = df['price'].replace('$', 1)
df['price'] = df['price'].replace('$$ - $$$',2)
df['price'] = df['price'].replace('$$$$', 3)

# Format the 'Address column' to only keep the area code

def get_area_code(string):
    try:
        return(re.search('\d{5}', string).group())
    except AttributeError:
        return(None)
    
df['area'] = df['address'].apply(lambda x: get_area_code(x))
df = df.drop('address', axis = 1)
df = df.dropna()

# Drop all areas that don't belong to Munich

not_munich = ['85356', '85640', '85540', '85551', '85646', '85737',
              '85757', '82194', '82041', '82194', '82041', '82067', '82031', '82049']
df = df[~df['area'].isin(not_munich)]

# Saving cleaned dataset to a csv file

df.to_csv('munich restaurants cleaned.csv')

# Load the cleaned dataset

df = pd.read_csv('munich restaurants cleaned.csv', index_col='Unnamed: 0')

# Group the dataframe by area code

df_by_area = df.groupby(by = 'area').mean()
df_by_area['number_reviews'] = df.groupby(by = 'area').sum()['number_reviews']
df_by_area = df_by_area.reset_index()
df_by_area['area'] = df_by_area['area'].astype(str)
df_by_area.columns = ['area', 'avg_rating', 'number_reviews', 'avg_price']

# Create a dataframe with geometrical data for all area codes
# Shapefile from https://www.suche-postleitzahl.org/downloads

area_shape_df = gpd.read_file('plz-gebiete.shp', dtype = {'plz': str})
area_shape_df = area_shape_df.drop('note', axis = 1)
area_shape_df = area_shape_df[area_shape_df['plz'].astype(str).str.startswith('8')]
area_shape_df.columns = ['area', 'geometry']

# Merge the dataframes and drop missing values

final_df = pd.merge(left = area_shape_df, right = df_by_area, on = 'area')
final_df = final_df.dropna()

# Apply a function to calculate the score of each area 
# https://math.stackexchange.com/questions/942738

p = final_df['avg_rating']
q = final_df['number_reviews']
Q = final_df['number_reviews'].median()

final_df['score'] = 0.5 * p + 2.5 * (1 - e**(-q / Q))

# Create plot to show the map
# Map from https://upload.wikimedia.org/wikipedia/commons/2/2d/Karte_der_Stadtbezirke_in_M%C3%BCnchen.png

plt.rcParams['figure.figsize'] = [48, 32]
img = plt.imread('munich map.png')

fig, ax = plt.subplots()

final_df.plot(ax = ax, zorder = 0, column = 'score', categorical = False, cmap='RdYlGn')
ax.imshow(img, zorder = 1, extent = [11.36, 11.725, 48.06, 48.250], alpha = 0.7)

red_patch = mpatches.Patch(color='#bb8995', label = 'Bad')
yellow_patch = mpatches.Patch(color='#d6d0b9', label = 'Okay')
green_patch = mpatches.Patch(color='#89ab9a', label = 'Good')
plt.legend(handles=[green_patch, yellow_patch, red_patch], facecolor = 'white', edgecolor='lightgrey',
           fancybox=True, framealpha=0.5, loc = 'right', bbox_to_anchor=(0.975, 0.925), ncol = 3, fontsize = 48)

ax.text(0.0375, 0.925, 'Where to Eat in Munich ?', fontsize = 80, weight = 'bold',
        transform = ax.transAxes, bbox = dict(facecolor = 'white', edgecolor = 'lightgrey', alpha = 0.6, pad = 25))
plt.annotate('Map based on 290.522 online ratings of 2,202 restaurants in Munich.',
             (0, 0), (0, -20), fontsize = 38, weight = 'bold', xycoords = 'axes fraction',
             textcoords='offset points', va = 'top')
plt.annotate('by: Gabriel Berardi', (0,0), (1960, -20), fontsize = 38, weight = 'bold', 
             xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.annotate('\nThe score of each area is calculated using the average rating and the total number of reviews:',
             (0, 0), (0, -70), fontsize = 38, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.annotate('score = 0.5 * avg_rating + 2.5 * (1 - e^( - number_reviews / median(number_reviews))',
             (0,0), (0, -165), fontsize = 38, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.annotate('(Formula by: Marc Bogaerts (math.stackexchange.com/users/118955))', (0, 0), (0, -220),
             fontsize = 32, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')

ax.set(facecolor = 'lightblue', aspect = 1.4, xticks = [], yticks = [])
plt.show()
