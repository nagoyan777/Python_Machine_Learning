#!/bin/env python

import re, os, numpy as np, pandas as pd
from utils import split_data, tokenizer tokenizer_porter, preprocessor

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pyprind
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from nltk.stem.porter import PorterStemmer

from utils import split_data, tokenizer tokenizer_porter, preprocessor, get_stopwords, preprocessor


def main():
    out_of_core()


def out_of_core():
    vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    doc_stream = stream_docs(path='./movie_data.csv')

    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test, y_test))
    clf = clf.partial_fit(X_test, y_test)


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
             yield text, label

def tokenizer(text):
    text = preprocessor(text)
    stop = get_stopwords()
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
