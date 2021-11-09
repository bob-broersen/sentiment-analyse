import hotel_reviews_tripAdviser_scraper as tripadvisor_scraper
import hotel_reviews_csv_export as csv_scraper
import mysql_connection as mysql
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

csv_table_name = 'Reviews_csv'
tripadvisor_table_name = 'Reviews_tripadvisor'


# csvdata = csv_scraper.scrape()
# mysql.put_in_df(csv_table_name, csvdata)

# tripadvisordata = tripadvisor_scraper.scrape()
# mysql.put_in_df(tripadvisor_table_name, tripadvisordata)

result = mysql.get_df()


# reviews = result.Review.values
# wordcloud = WordCloud(
#     width = 3000,
#     height = 2000,
#     background_color = 'black',
#     stopwords = STOPWORDS).generate(str(reviews))
# fig = plt.figure(
#     figsize = (40, 30),
#     facecolor = 'k',
#     edgecolor = 'k')
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=1000, stop_words=STOPWORDS, token_pattern=r'(?u)\b[A-Za-z]+\b').fit(result.Review)
tfidf = vect.transform(result.Review)
X=pd.DataFrame(tfidf.toarray(), columns=vect.get_feature_names())
y = result.Label.apply(lambda x: 1 if x == 'Positive review' else 0)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import pickle
import time

def test_model(model, test_set, check_set):
    t0 = time.time()
    predicted_set = model.predict(test_set)
    print('Accuracy score test set: ', accuracy_score(check_set, predicted_set))
    print('Confusion matrix test set: \n', confusion_matrix(check_set, predicted_set))
    print(classification_report(predicted_set,check_set))
    t1 = time.time()
    print("time to test: " + str(t1 - t0))

def build_model(model_type, X_train, y_train):
    t0 = time.time()
    model = model_type.fit(X_train, y_train)
    t1 = time.time()
    print("time to build model: " + str(t1 - t0))  
    return model

def live_prediction(model, vectorizer, text):
    if vectorizer != None:
        test = vectorizer.transform([text]).toarray()
    else:
        test = [text]
    pred = model.predict(test)
    print("Positive review" if pred[0] == 1 else "Negative review")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = build_model(LogisticRegression(), X_train, y_train)
test_model(log_reg, X_test, y_test)
pickle.dump(log_reg, open('log_reg.pkl', 'wb'))

naiveBayes  = build_model(MultinomialNB(), X_train, y_train)
test_model(naiveBayes, X_test, y_test)
pickle.dump(naiveBayes, open('naiveBayes.pkl', 'wb'))

adaboost  = build_model(AdaBoostClassifier(), X_train, y_train)
test_model(adaboost, X_test, y_test)
pickle.dump(adaboost, open('adaboost.pkl', 'wb'))




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

t0 = time.time()
pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ]
)

parameters = {
    "vect__max_df": (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    "vect__ngram_range": ((1, 1), (1, 2)),  
    'tfidf__use_idf': (True, False),
}

grid = build_model(GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1), result.Review, y)
test_model(grid, result.Review, y)
print("Best score: %0.3f" % grid.best_score_)
print("Best parameters set:")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
pickle.dump(grid, open('grid_search.pkl', 'wb'))

loaded_grid = pickle.load(open('grid_search.pkl', 'rb'))
test_model(loaded_grid, result.Review, y)

loaded_log_reg = pickle.load(open('log_reg.pkl', 'rb'))
test_model(loaded_grid, X_test, y_test)

loaded_naive_bayes = pickle.load(open('naiveBayes.pkl', 'rb'))
test_model(loaded_grid, X_test, y_test)

loaded_ada = pickle.load(open('adaboost.pkl', 'rb'))
test_model(loaded_ada, X_test, y_test)


live_prediction(loaded_grid, None, "this was a really bad hotel")
live_prediction(loaded_naive_bayes, vect, "This was a really good hotel")