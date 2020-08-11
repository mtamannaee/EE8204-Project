# Mahtab Tamannaee 
# 500634850
# EE8204

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
nltk.download('stopwords')
import gensim
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidatasetVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


DS_SCHEME = ["Sentiment", "Ids", "Date", "Flag", "User", "text"]
dataset = pd.read_csv(os.path.join("./data/dataset.csv"), encoding ="ISO-8859-1" , names = DS_SCHEME)


polarity_score = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def translate_polarity(label):
    return polarity_score[int(label)]

dataset.Sentiment = dataset.Sentiment.apply(lambda x: translate_polarity(x))

Sentiment_cnt = Counter(dataset.Sentiment)

plt.figure(figsize=(9,5))
plt.bar(Sentiment_cnt.keys(), Sentiment_cnt.values())
plt.title("Corpus Polarity Distribuition")

## data preprocessing -----------------------------------------------------------------------

stop_words = set(stopwords.words("english"))
stop_words.remove("not")
stop_words.remove("no")
stop_words= list(stop_words)
stemmer = SnowballStemmer("english")
SEQUENCE_LENGTH = 300

def fixNot_text(text):
	fixed_text = []
	for word in text:
		if re.search("n't", word):
			fixed_text.append(word.split("n't")[0])
			fixed_text.append("not")
		else:
			fixed_text.append(word)
	return " ".join(fixed_text)


CHARACTERS = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text, stem=False):
	fixNot_text(text)
    text = re.sub(CHARACTERS, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

dataset.text = dataset.text.apply(lambda x: preprocess(x))

dataset_train, dataset_test = train_test_split(dataset, test_size=1-0.8, random_state=20)
print("TRAIN : {} , TEST : {}".format( len(dataset_train), len(dataset_test))

documents = [_text.split() for _text in dataset_train.text] 
      
## Word2Vec ------------------------------------------------------------------------------------
w2v_model = gensim.models.word2vec.Word2Vec(size = 300, window = 10, min_count = 10, workers = 8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", len(words))

w2v_model.train(documents, total_examples=len(documents), epochs= 32)
      
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words count : ", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(dataset_train.text), maxlen = 300)
x_test = pad_sequences(tokenizer.texts_to_sequences(dataset_test.text), maxlen = 300)

#labels : sentiment polarity
labels = dataset_train.Sentiment.unique().tolist()
labels.append("NEUTRAL")
print(labels)

encoder = LabelEncoder()
encoder.fit(dataset_train.Sentiment.tolist())

y_train = encoder.transform(dataset_train.Sentiment.tolist())
y_test = encoder.transform(dataset_test.Sentiment.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)

print("x_train", x_train.shape) # x_train (1280000, 300)
print("y_train", y_train.shape) # y_train (1280000, 1)
print("x_test", x_test.shape)   # x_test (320000, 300)
print("y_test", y_test.shape)   # y_test (320000, 1)

# word2vec Matrix for Embedding layer  -----------------------------------------------------------------
W2V_VEC_SIZE = 300    
print("W2V_VEC_SIZE : ",W2V_VEC_SIZE)
print("vocab_size : ",vocab_size)

wv_em_matrix = np.zeros((vocab_size, W2V_VEC_SIZE))

for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    wv_em_matrix[i] = w2v_model.wv[word]
print(wv_em_matrix.shape)


# Nueral Network Model  ---------------------------------------------------------------------------
#Embedding layer 
embedding_layer = Embedding(vocab_size, W2V_VEC_SIZE, weights=[wv_em_matrix], input_length= 300, trainable=False)

# LSTM Model
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

history = model.fit(x_train, y_train,batch_size=1024, epochs = 10, validation_split=0.1, verbose=1, callbacks=callbacks)
      
score = model.evaluate(x_test, y_test, batch_size = 1024)
print("Accuracy:",score[1])
print("Loss:",score[0])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training Set Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Set Loss')
plt.title('Training and validation Loss')
plt.legend()
plt.show()


SENTIMENT_THRESHOLDS = (0.4, 0.7)
def translate_polarity(score, include_neutral=True):
    if include_neutral:        
        label = "NEUTRAL"
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = "NEGATIVE"
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = "POSITIVE"
        return label
    else:
        return "NEGATIVE" if score < 0.5 else "POSITIVE"

def predict(text, include_neutral=True):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = model.predict([x_test])[0] # Predict
    label = translate_polarity(score, include_neutral=include_neutral)  # sentiment extraction
    return {"label": label, "score": float(score)}


y_pred_1d = []
y_test_1d = list(dataset_test.Sentiment)
scores = model.predict(x_test, verbose=1, batch_size = 8000)
y_pred_1d = [translate_polarity(score, include_neutral=False) for score in scores]

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=17)
    plt.yticks(tick_marks, classes, fontsize=17)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual Label', fontsize=17)
    plt.xlabel('Predicted Label', fontsize=17)
    
cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=dataset_train.Sentiment.unique(), title="Confusion Matrix")
plt.show()

print(classification_report(y_test_1d, y_pred_1d))   # Classification Final Results
accuracy_score(y_test_1d, y_pred_1d)   # Accuracy 

