import json
import re

import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from scipy.sparse import hstack
import nltk
import xgboost as xgb


def preprocess(text):
    # make 'less' equal to '\nless'
    text = [i.replace("\n", " ") for i in text]
    text = [i.replace("\r", " ") for i in text]

    # If this punctuation is removed as is, words will be incorrectly joined together
    text = [i.replace("--", " -- ") for i in text]

    text = [re.sub(' +', ' ', i) for i in text] # multiple spaces -> single space

    text = [i.lower() for i in text] # lowercase
    return text

def main():
    nltk.download('averaged_perceptron_tagger')

    # Read in the data
    f = open("data/gutenberg-paragraphs.json")
    data = json.load(f)
    text = [i["text"] for i in data]
    labels = np.array([i["austen"] for i in data])

    # Preliminary analysis
    print("Number of rows:", labels.size)
    print("Positive-negative ratio: {r:.3f}".format(r = np.mean(labels)))

    text = preprocess(text)

    kf = KFold(n_splits=5, shuffle=True)
    f1_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(text)):
        print(f"\nFold {i + 1}:")
        X_train = [text[i] for i in train_index]
        X_test = [text[i] for i in test_index]
        y_train = np.array([labels[i] for i in train_index])
        y_test = np.array([labels[i] for i in test_index])

        # Tokenize
        # Words
        word_vect = TfidfVectorizer(ngram_range=(1,2), min_df=30, max_features=10000)
        word_vect.fit(X_train)
        X_train_word = word_vect.transform(X_train)
        X_test_word = word_vect.transform(X_test)

        # Characters - authors can be identified by their use of particular punctuatuion
        char_vect = TfidfVectorizer(ngram_range=(1,2), min_df=30, max_features=10000, analyzer='char', preprocessor=None)
        char_vect.fit(X_train)
        X_train_char = char_vect.transform(X_train)
        X_test_char = char_vect.transform(X_test)

        # Part of speech tags - generalise about sentence structure without concern for particular words
        X_train_pos_tags = [" ".join([j[1] for j in nltk.pos_tag(i.split(" "))]) for i in X_train]
        X_test_pos_tags = [" ".join([j[1] for j in nltk.pos_tag(i.split(" "))]) for i in X_test]
        pos_vect = TfidfVectorizer(ngram_range=(1,2), min_df=30, max_features=10000)
        pos_vect.fit(X_train_pos_tags)
        X_train_pos = word_vect.transform(X_train_pos_tags)
        X_test_pos = word_vect.transform(X_test_pos_tags)

        X_train_counts = hstack([X_train_word, X_train_char, X_train_pos])
        X_test_counts = hstack([X_test_word, X_test_char, X_test_pos])

        # Create and fit the model
        # XGBoost on word and character unigrams and bigrams
        model = xgb.XGBClassifier(n_estimators=256, max_depth=4, learning_rate=0.3, objective='binary:logistic')
        model.fit(X_train_counts, y_train)

        print("\nTrain set")
        pred = model.predict(X_train_counts)
        print("Accuracy: {a:.3f}".format(a = accuracy_score(y_train, pred)))
        print("Precision: {p:.3f}".format(p = precision_score(y_train, pred)))
        print("Recall: {r:.3f}".format(r = recall_score(y_train, pred)))
        print("F1: {f:.3f}".format(f = f1_score(y_train, pred)))

        print("\nTest set")
        pred = model.predict(X_test_counts)
        print("Accuracy: {a:.3f}".format(a = accuracy_score(y_test, pred)))
        print("Precision: {p:.3f}".format(p = precision_score(y_test, pred)))
        print("Recall: {r:.3f}".format(r = recall_score(y_test, pred)))

        f1 = f1_score(y_test, pred)
        print("F1: {f:.3f}".format(f = f1))
        f1_scores.append(f1)

        # Manual evaluation of errors
        if i == 0:
            print("\nFalse positives:")
            false_positives = np.where(np.logical_and(pred == 1, y_test == 0))
            false_positives_sample = np.random.choice(false_positives[0], size=10, replace=False)
            for idx in false_positives_sample:
                print(X_test[idx])

            print("\nFalse negatives:")
            false_negatives = np.where(np.logical_and(pred == 0, y_test == 1))
            false_negatives_sample = np.random.choice(false_negatives[0], size=10, replace=False)
            for idx in false_negatives_sample:
                print(X_test[idx])

    avg_f1 = np.mean(f1_scores)
    print("Average F1 score: {f:.3f}".format(f = avg_f1))

if __name__ == "__main__":
    main()