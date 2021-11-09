import pandas as pd
from bs4 import BeautifulSoup
import requests

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
base_url = 'https://www.tripadvisor.com/'

all_reviews = []
AMOUNT_REVIEWS_HOTEL = 10

def get_soup(url):
    r = requests.get(url, headers=headers)#, proxies=proxies)
    content = r.content
    return BeautifulSoup(content)

def get_hotels(no_page):
    page_request_text = ''

    if (no_page > 1):
        page_request_text = '-oa' + str(no_page * 30)
    
    soup = get_soup( base_url + 'Hotels-g186338' + page_request_text + '-London_England-Hotels.html')

    for d in soup.findAll('a', href=True, attrs={'class':'property_title prominent'}):
        all_reviews.extend(get_review_from_hotels(d['href']))

def get_review_from_hotels(hotel_url):
    reviews_hotel = []
    next_page = True
    to_replace = "Reviews-"
    review_counter = 0
    while(next_page):
        review_counter += 5
        soup = get_soup( base_url + hotel_url)
        review = []

        for review_div in soup.findAll('div', attrs={'class':'cWwQK MC R2 Gi z Z BB dXjiy'}):
            print('going trough review')
            review_q = review_div.find('q', attrs={'class':'XllAv H4 _a'})
            review_text = review_q.find('span').text
            review.append(review_text)
            review_bubble = review_div.find('span', attrs={'class':'ui_bubble_rating'})['class'][1]
            if review_bubble == 'bubble_40':
                review_label = 'Positive'
            elif review_bubble == 'bubble_50':
                review_label = 'Positive'
            else:
                review_label = 'Negative'
            review.append(review_label)
            reviews_hotel.append(review)
            review = []

        replacement = 'Reviews-or' + str(review_counter) + '-'
        hotel_url.replace(to_replace, replacement)
        to_replace = replacement

        if len(reviews_hotel) >= AMOUNT_REVIEWS_HOTEL:
            return reviews_hotel
            
        if soup.find('span', attrs={'class':'ui_button nav next primary disabled'}):
            next_page = False


def scrape():
    get_hotels(1)
    df = pd.DataFrame(all_reviews, columns=['Review','Label'])
    return df
