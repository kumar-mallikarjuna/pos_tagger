from tensorflow import keras
import nltk
from sklearn.feature_extraction import DictVectorizer

import numpy as np
import pickle

# VOCABULARY SIZE
m = 10000

def feature_extractor(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence),
        'is_capitalized': sentence[index] == sentence[index].capitalize(),
        'is_all_caps': sentence[index] == sentence[index].upper(),
        'is_all_lower': sentence[index] == sentence[index].lower(),
        'previous_word': '' if index == 0 else sentence[index-1],
        'next_word': '' if index == len(sentence)-1 else sentence[index+1],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'has_hyphen': '-' in sentence[index],
        'is_number': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:] != sentence[index][1:].lower()
    }

def untag(x):
    return [a[0] for a in x]

def transform(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(feature_extractor(untag(tagged), index))
            y.append({'label':tagged[index][1]})

    return X, y

training_sentences = nltk.corpus.treebank.tagged_sents()    # Loading training sentences from Treebank Corpus

x, y = transform(training_sentences)    # Extract features from training_sentences

vectorizer_x = DictVectorizer(sparse=False)     # Vectorizer for Input
vectorizer_x.fit(x[:m])
x = vectorizer_x.transform(x[:m])

vectorizer_y = DictVectorizer(sparse=False)     # Vectorizer for Target
vectorizer_y.fit(y[:m])
y = vectorizer_y.transform(y[:m])

def train(name, epochs=3):
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=x[0].shape),
        keras.layers.Dense(y[0].size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, validation_data=(x,y), epochs=epochs)

    model.save("model")

def run(model_name, sentence):
    model = keras.models.load_model(model_name)
    sentence = nltk.word_tokenize(sentence)

    X = []

    for i in range(len(sentence)):
        X.append(feature_extractor(sentence, i))

    X = vectorizer_x.transform(X)
    Y = model.predict(X)
    d = vectorizer_y.inverse_transform(Y)

    l = []

    for i in range(len(d)):
        max_key = 'label=$'
        
        for k, v in d[i].items():
            if(k.startswith("label=")):
                if(v > d[i][max_key]):
                    max_key = k

        l.append((sentence[i], max_key[max_key.index('=')+1:]))

    return l

print(run("./model", "Outside my mind, nothing is real."))
