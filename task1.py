import html
import json

import keras.src.losses
import pandas as pd
import re
import numpy as np
from keras.src.saving.legacy.saved_model.load import models_lib
from keras.src.saving.legacy.saved_model.serialized_attributes import metrics
from sklearn.metrics import classification_report
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
import json
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.legacy_tf_layers.core import dropout

def custom_standardize(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text

def loadData(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def dataPrep(raw_data):
    data = html.unescape(raw_data)
    data_new = re.sub(r'\s+', ' ', data)
    data_output = ' '.join(data_new.split())
    return data_output

def getFeatures(input_list):
    post = input_list['postText'][0]
    title = input_list.get('targetTitle') or ''
    paragraph = input_list.get('targetParagraphs') or []

    paragraph = ' '.join(paragraph)
    # head_para = ' '.join(paragraph[:3]) # top 3 sentences
    # tail_para = ''

    # if len(paragraph) > 3:
    #     tail_para = ' '.join(paragraph[-3:])
    # else:
    #     tail_para = ''
    # output_feature = ' '.join([post, title, head_para, tail_para])
    output_feature = ' '.join([post, title, paragraph])
    return dataPrep(output_feature)

def featuresList(input_file):
    output_list = []
    for item in input_file:
        output_list.append(getFeatures(item))
    return output_list

def getLable(input_list):
    lable_list = []
    id_dict = {
        'phrase': 0,
        'passage': 1,
        'multi': 2
    }
    for item in input_list:
        lable = item['tags'][0]
        numerical_lable = id_dict[lable]
        lable_list.append(numerical_lable)
    return np.array(lable_list, dtype='int32')

def main():
    # load data
    train = loadData('train.jsonl')
    val = loadData('val.jsonl')
    test = loadData('test.jsonl')

    # preprocessing data
    Xtrain = featuresList(train)
    Xval = featuresList(val)
    Xtest = featuresList(test)

    ytrain = np.array(getLable(train), dtype='int32')
    yval = np.array(getLable(val), dtype='int32')



    input_vector = layers.TextVectorization(standardize=custom_standardize,
                                            max_tokens=50000, output_sequence_length=1000,
                                            split='whitespace')
    input_vector.adapt(Xtrain)

    model = models.Sequential()
    model.add(input_vector)
    model.add(layers.Embedding(
        input_dim=len(input_vector.get_vocabulary()), output_dim=128, mask_zero=True
    ))
    model.add(layers.SpatialDropout1D(0.5))
    model.add(layers.Conv1D(filters=256, kernel_size=5))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.55))
    model.add(layers.Conv1D(filters=128, kernel_size=5))
    model.add(layers.LeakyReLU())
    model.add(layers.SpatialDropout1D(0.4))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Bidirectional(layers.LSTM(16, return_sequences=False)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.55))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stop = callbacks.EarlyStopping(patience=6, restore_best_weights=True)

    model.fit(
        x=np.array(Xtrain), y=ytrain,
        validation_data=(np.array(Xval), yval), epochs=90, batch_size=8,
        callbacks=[early_stop,], verbose=1, shuffle=True
    )

    ytest_pred = model.predict(np.array(Xtest), batch_size=32).argmax(axis=1)
    ytest_pred_text = []
    for result in ytest_pred:
        if result == 0:
            ytest_pred_text.append('phrase')
        elif result == 1:
            ytest_pred_text.append('passage')
        else:
            ytest_pred_text.append('multi')

    testID = []
    for item in test:
        testID.append(item['id'])

    df_test = pd.DataFrame({
        'id': testID,
        'spoilerType': ytest_pred_text
    })
    df_test.to_csv('task 1 solution.csv')

    ytrain_pred = model.predict(np.array(Xtrain), batch_size=32, verbose=0).argmax(axis=1)

    yval_pred = model.predict(np.array(Xval), batch_size=32, verbose=0).argmax(axis=1)

    model.save('Task1Model.keras')
    cm1 = confusion_matrix(yval, yval_pred)
    print(cm1)
    print(classification_report(yval, yval_pred, target_names=['0', '1', '2']))

    cm2 = confusion_matrix(ytrain, ytrain_pred)
    print(cm2)
    print(classification_report(ytrain, ytrain_pred, target_names=['0', '1', '2']))

if __name__ == '__main__':
    main()







