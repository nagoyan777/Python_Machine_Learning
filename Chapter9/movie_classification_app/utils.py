#!/bin/env python
import os, re, numpy as np, pandas as pd
from nltk.stem.porter import PorterStemmer
import pyprind



def get_stopwords():
    import nltk
    from nltk.corpus import stopwords
    try:
        stop = stopwords.words('english')
    except Exception as p:
        nltk.download('stopwords')
        stop = stopwords.words('english')
    return stop


def split_data(df, review='review', sentiment='sentiment',ntrain=25000, ntest=25000):
    X_train = df.loc[:ntrain, review].values
    y_train = df.loc[:ntest, sentiment].values
    X_test = df.loc[:ntrain, review].values
    y_test = df.loc[:ntest, sentiment].values
    return X_train, y_train, X_test, y_test


#def tokenizer(text):
#    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    return text


def read_data(filename='movie_data.csv'):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = _read_aclImdb()
        df = _shuffle_data(df)
        df.to_csv('./movie_data.csv', index=False)
    return df


def _shuffle_data(df):
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
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
