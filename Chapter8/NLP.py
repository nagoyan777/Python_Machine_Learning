#!/bin/env python
import os
import re
import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import pyprind
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from utils import split_data, tokenizer, tokenizer_porter, preprocessor, get_stopwords


def main():
    bagwords()


def bagwords():
    df = read_data()
    count, bag, docs = bug_of_words(df)
    tfidf = TfidfTransformer()
    np.set_printoptions(precision=2)

    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
    stop = get_stopwords()

    df['review'] = df['review'].apply(preprocessor)
    X_train, y_train, X_test, y_test = split_data(df, ntrain=25000, ntest=25000)

    tfidf = TfidfVectorizer(
        strip_accents=None, lowercase=False, preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words':[stop, None],
                   'vect__tokenizer':[tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer':[tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]}]
    lr_tfidf = Pipeline(
        [('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(
        lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)

    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

    dest = os.path.join('movieclassifier', 'pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)

def bug_of_words(df):
    count = CountVectorizer()
    docs = np.array([
                    'The sun is shining',
                    'The weather is sweet',
                    'The sun is shining and the weather is sweet'])
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    return count, bag, docs


#--> moved to utils.py
#def split_data(df, review='review', sentiment='sentiment',ntrain=25000, ntest=25000):
#    X_train = df.loc[:ntrain, review].values
#    y_train = df.loc[:ntest, sentiment].values
#    X_test = df.loc[:ntrain, review].values
#    y_test = df.loc[:ntest, sentiment].values
#    return X_train, y_train, X_test, y_test


#def tokenizer(text):
#    return text.split()


#def tokenizer_porter(text):
#    porter = PorterStemmer()
#    return [porter.stem(word) for word in text.split()]


# def preprocessor(text):
#     text = re.sub('<[^>]*>', '', text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = re.sub('[\W]+', ' ', text.lower()) + \
#         ' '.join(emoticons).replace('-', '')
#    return text


def shuffle_data(df):
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    return df


def read_data(filename='movie_data.csv'):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = _read_aclImdb()
        df = shuffle_data(df)
        df.to_csv('./movie_data.csv', index=False)
    return df


def _read_aclImdb(dirname='aclImdb'):
    pbar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = './%s/%s/%s' % (dirname, s, l)
            for file_ in os.listdir(path):
                with open(os.path.join(path, file_), 'r') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']
    return df


def get_data(filename):
    pass


if __name__ == '__main__':
    main()
