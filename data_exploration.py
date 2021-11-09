import pandas as pd
df = pd.read_csv("Hotel_Reviews.csv")
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px


df.head()
reviews_positive = df.Positive_Review.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(reviews_positive))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

reviews_negative = df.Negative_Review.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(reviews_negative))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

reviews = reviews_positive + reviews_negative
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(reviews))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

reviews_hotel = df[['Hotel_Name', 'Average_Score']].groupby([pd.Grouper(key='Hotel_Name')]).agg('count').reset_index()
reviews_hotel.head()
fig = px.bar(reviews_hotel, x='Hotel_Name', y='Average_Score', labels={
    'Hotel_Name':'Hotel',
    'Average_Score':'Amount of reviews'
    }, title='Amount of reviews per hotel')
fig.show()

reviews_hotel = df[['Reviewer_Nationality', 'Average_Score']].groupby([pd.Grouper(key='Reviewer_Nationality')]).agg('count').reset_index()
reviews_hotel.head()
fig = px.bar(reviews_hotel, x='Reviewer_Nationality', y='Average_Score', labels={
    'Reviewer_Nationality':'Nationality',
    'Average_Score':'Amount of reviews'
    }, title='Amount of reviews per nationality')
fig.show()