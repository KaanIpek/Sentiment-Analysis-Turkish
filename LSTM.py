import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from random import sample,random
import pandas as pd
import requests
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import gzip
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
from numba import jit, cuda, numba

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12} )
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)




numba.cuda.jit
nltk.download('stopwords')
nltk.download('punkt')
name = "serve.pos"
# name = "interest.pos"
# name = "hard.pos"
# name = "line.pos"
downloadRoot = "https://www.batikanor.com/data/senseval/"
dataUrl = downloadRoot + name
labelTag = "senseid="
tag = "context"
reg_str = "<" + tag + ">(.*?)</" + tag + ">"
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def downloadSenseval():
    print("Loading " + name + " data...")
    myFile =  requests.get(dataUrl)
    print("Loaded")
    return myFile.text
def getLabels():
    """
    Extracts the labels from label tags specified on resp. dataset
    """


    labels = []

    for line in txt.split('\n'):
        for w in line.split():
            if labelTag in w and w.endswith("\"/>"):
                labels.append(w[w.find('"') + 1 : w.rfind('"')])
    # print(labels, len(labels), sep="\n")

    # sort class types, name the first one on alphabetical order: 0
    classNames =  sorted(set(labels))
    ys = dict((j, i) for i, j in enumerate(classNames))
    y = np.array([ys[label] for label in labels]) # interest_1 -> 0, ...
    # print(y, labels)
    return y, classNames
txt = downloadSenseval()
strs = re.findall(reg_str, txt, re.DOTALL) # DOTALL matches '.' for ALL characters ( including '\n' )
stop_words = set(stopwords.words('english'))

texts = []
print_stuff=1
for ctx in strs:
# for sentence in sentence_in_doc:
    ctx_content = (re.sub('<[^>]*>', '', ctx)) # removes all pos tags
    # print(ctx_content)
    ctx_no_stopwords = [w for w in ctx_content.split() if w.lower() not in stop_words]
    # print(ctx_no_stopwords)
    ctx_clean = str(' ').join([w for w in ctx_no_stopwords if (w not in string.punctuation and w.isalpha() and len(w)>1) ])

    texts.append(ctx_clean)
if print_stuff:
    len(texts), texts

y, classNames = getLabels()
print(len(y), y)
num_classes = len(classNames)
print(len(classNames), classNames)
print(len(texts) == len(y))

limit = -1
sampling = "direct"
if limit != -1:
    if sampling == "direct":
        texts = texts[:limit]
        y = y[:limit]
    elif sampling == "random":
        texts, y = zip(*sample(list(zip(texts, y)), limit))
print(texts[0], y[0])
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=test_size, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_trains = tokenizer.texts_to_sequences(X_train)
X_tests = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

#look at example
print(X_train[2])
print(X_trains[2])

lens = np.array([len(item) for item in X_trains])
#nn'ne verirken review uzunluklarını eşitliyoruz
maxlen = lens.max()
print(maxlen)
X_trains = pad_sequences(X_trains, padding='post', maxlen=maxlen)
X_tests = pad_sequences(X_tests, padding='post', maxlen=maxlen)

print(X_trains[0, :])
#glove.6B.50d
from tensorflow.keras.utils import to_categorical

# Convert the labels to one_hot_category values
y_train_1hot = to_categorical(y_train, num_classes = 4)
y_test_1hot = to_categorical(y_test, num_classes = 4)
print(y_train)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, 'r', encoding='UTF-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix('glove.6B.50d.txt',tokenizer.word_index, embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


embedding_vector_features=50

# model=Sequential()
#
#
# model.add(Embedding(input_dim=vocab_size, output_dim=50,  weights=[embedding_matrix], input_length=maxlen, trainable=True))
#
# # model.add(Bidirectional(LSTM(128,activation='relu',return_sequences=True)))
# #
# # model.add(Dropout(0.2))
#
# model.add(Bidirectional(LSTM(128,activation='relu')))
#
# model.add(Dropout(0.2))#perfect value
#
# for units in [128,128,64,32]:
#
#     model.add(Dense(units,activation='relu'))
#
#     model.add(Dropout(0.2))
#
# model.add(Dense(32,activation='relu'))
#
# model.add(Dropout(0.2))
#
# model.add(Dense(4,activation='softmax'))
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#
# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['categorical_accuracy'])
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.Conv1D(filters=50, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(units=50, activation='relu'))
model.add(layers.Dense(units=4, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()
print(model.summary())
print(X_trains.shape)
print(X_tests.shape)
print(y_train_1hot.shape)
print(y_test_1hot.shape)

history=model.fit(X_trains,y_train_1hot,validation_data=(X_tests,y_test_1hot),epochs=200,batch_size=256)
results = model.evaluate(X_tests,y_test_1hot)
y_pred = model.predict(X_tests)
y_pred_cat=y_pred.argmax(axis=1)
print(y_pred_cat)
print(results)
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred_cat,average='macro'))


acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# X_trainraw=texts[:4000]
# X_testraw=texts[4000:]
# y_trainraw=y[:4000]
# y_testraw=y[4000:]
# vectorize data
# vect = CountVectorizer()
# X_train = vect.fit_transform(X_train)
# X_test = vect.transform(X_test)

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(X_trainraw)
# x_test=vectorizer.transform(X_testraw)
# X_train = X.toarray()
# X_test = x_test.toarray()
#
# x_traindf=pd.DataFrame(X_train,columns=vectorizer.get_feature_names())
# x_testdf=pd.DataFrame(X_test,columns=vectorizer.get_feature_names())
#
# i=0
# xtrainlist=[]
# for text in X_trainraw:
#     row = []
#
#
#     for word in text.split():
#         row.append(x_traindf[word][i])
#
#     xtrainlist.append(row)
#     i += 1
# xtrainarray=boolean_indexing(xtrainlist,0.0)
#
#
# xtestlist=[]
# i=0
# for text in X_testraw:
#     row = []
#     for word in text.split():
#         try:
#             row.append(x_testdf[word][i])
#         except:
#             row.append(0.0)
#
#     xtestlist.append(row)
#     i += 1
# xtestarray=boolean_indexing(xtestlist,0.0)
# print(xtestarray[0])
