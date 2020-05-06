# You can find the blog post about this code at: https://www.datasciencecoffee.com/2020-scrape-book-shop/

# Importing needed packages

import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# The following class enables us to access different elements of our crawled books

class CrawledBooks():
    def __init__(self, title, price, rating, image, available):
        self.title = title
        self. price = price
        self.rating = rating
        self.image = image
        self.available = available
        
# The following class defines the crawler itself

class BookCrawler():
    def fetch(self):
        
        url = 'http://books.toscrape.com/'
        r = requests.get(url)
        doc = BeautifulSoup(r.text, "html.parser")
        books = []
        
        # The following while-loop is executed until the last page has been reached
        while doc.select('.next'):
            
            # We set a break of 1 second in between each request and print the URL that is currently scraped
            time.sleep(1)
            print(url)
            
            url = urljoin(url, doc.select_one('.next a').attrs['href'])
            r = requests.get(url)
                                  
            for element in doc.select('.product_pod'):
                
                title = element.select_one('h3').text
                price = element.select_one('.price_color').text[2:]
                rating = element.select_one('p').attrs['class'][1]
                image = urljoin(url, element.select_one('.thumbnail').attrs['src'])
                available = element.select_one('.instock').text[15:23]
                
                crawled_books = CrawledBooks(title, price, rating, image, available)
                books.append(crawled_books)
            
            try:
                doc = BeautifulSoup(r.text, "html.parser")
                
            except:
                print('\n Crawling complete!')
                break
                
        return books

crawler = BookCrawler()
scraped_books = crawler.fetch()

# Next, we save the data to variables as a list, using list comprehension

all_titles = [i.title for i in scraped_books]
all_prices = [i.price for i in scraped_books]
all_ratings = [i.rating for i in scraped_books]
all_images = [i.image for i in scraped_books]
all_available = [i.available for i in scraped_books]

# At last, we can assemble the gathered data in a pandas data frame and save the result to a CSV or Excel file

df = pd.DataFrame(
    {'Title': all_titles,
     'Price (£)': all_prices,
     'Rating': all_ratings,
     'Image': all_images,
     'In Stock ?' : all_available
    })

df.to_csv('Scraped Books.csv')
df.to_excel('Scraped Books.xlsx')
