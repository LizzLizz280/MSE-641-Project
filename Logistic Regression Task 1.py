import numpy as np
import pandas as pd
import json
import html
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def loadData(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def clean(text):
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def joinFields(item):
    txt = ' '.join(item['postText'])
    txt += ' ' + item.get('targetTitle', '')
    txt += ' ' + ' '.join(item.get('targetParagraphs', []))
    return clean(txt)

def loadToTrainX(input):
    X = []
    for item in input:
        row = joinFields(item)
        X.append(row)
    return X

def loadToTrainY(input):
    y = []
    for item in input:
        label = item['tags'][0]
        y.append(label)
    return y

if __name__ == "__main__":
    train = loadData('train.jsonl')
    val = loadData('val.jsonl')

    Xtrain = loadToTrainX(train)
    ytrain = loadToTrainY(train)
    Xval = loadToTrainX(val)
    yval = loadToTrainY(val)

    vectorized = TfidfVectorizer(ngram_range=(1, 2),
                                 min_df=3, max_df=0.9,
                                 sublinear_tf=True)

    Xtr = vectorized.fit_transform(Xtrain)
    Xva = vectorized.transform(Xval)

    lr = LogisticRegression(max_iter=100, class_weight='balanced')
    lr.fit(Xtr, ytrain)

    joblib.dump((vectorized, lr), 'logistic_reg.pkl')

    yval_pred = lr.predict(Xva)

    cm = confusion_matrix(yval, yval_pred, labels=lr.classes_)
    print(cm)
