import pandas as pd
import mysql_connection as mysql

df = pd.read_csv("Hotel_Reviews.csv")

columnNameReview = "Review"

positiveReviews = (
    df.loc[:,['Positive_Review']]
    .assign(Label="Positive review")
    .rename(columns={"Positive_Review": columnNameReview})

)

negativeReviews = (
    df.loc[:,['Negative_Review']]
    .assign(Label='Negative review')
    .rename(columns={"Negative_Review": columnNameReview})

)

result = (
    pd.concat([positiveReviews, negativeReviews])
    .sample(frac=1, random_state= 2).reset_index(drop=True)
)

def scrape():
    return result