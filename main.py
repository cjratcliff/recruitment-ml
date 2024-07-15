import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.sparse import hstack
import xgboost as xgb

# Read in the data
f = open("data/gutenberg-paragraphs.json")
data = json.load(f)
text = [i["text"] for i in data]
labels = np.array([i["austen"] for i in data])

# Preliminary analysis
print("Number of rows:", labels.size)
print("Positive-negative ratio: {r:.3f}".format(r = np.mean(labels)))

# Preprocessing

# make 'less' equal to '\nless'
text = [i.replace("\n", " ") for i in text]
text = [i.replace("\r", " ") for i in text]

# If this punctuation is removed as is, words will be incorrectly joined together
text = [i.replace("--", " -- ") for i in text]

text = [re.sub(' +', ' ', i) for i in text] # multiple spaces -> single space

text = [i.lower() for i in text] # lowercase

# Create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=.2)

#### Baseline model (XGBoost on a bag of words and characters) ####

# Tokenize
count_vect = CountVectorizer(ngram_range=(1,1), min_df=10)
count_vect.fit(X_train)
X_train_word = count_vect.transform(X_train)
X_test_word = count_vect.transform(X_test)

char_vect = CountVectorizer(ngram_range=(1,1), min_df=10, analyzer='char', preprocessor=None)
char_vect.fit(X_train)
X_train_char = char_vect.transform(X_train)
X_test_char = char_vect.transform(X_test)

X_train_counts = hstack([X_train_word, X_train_char])
X_test_counts = hstack([X_test_word, X_test_char])

# tfidf_vect = TfidfVectorizer(min_df=10)
# tfidf_vect.fit(X_train)
# X_train_counts = tfidf_vect.transform(X_train)
# X_test_counts = tfidf_vect.transform(X_test)

# Create and fit the model
model = xgb.XGBClassifier(n_estimators=32, max_depth=12, learning_rate=1, objective='binary:logistic')
model.fit(X_train_counts, y_train)

# Evaluate the baseline
pred = model.predict(X_test_counts)
print("\nBaseline: XGBoost on a bag of words")
print("Test set")
print("Accuracy: {a:.3f}".format(a = accuracy_score(y_test, pred)))
print("Precision: {p:.3f}".format(p = precision_score(y_test, pred)))
print("Recall: {r:.3f}".format(r = recall_score(y_test, pred)))
print("F1: {f:.3f}".format(f = f1_score(y_test, pred)))