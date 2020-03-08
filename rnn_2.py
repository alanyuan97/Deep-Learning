import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten, Add, Lambda
from keras.layers import LSTM, CuDNNLSTM
from keras.datasets import imdb
from keras.utils import np_utils
from gensim.models import word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.layers.wrappers import Bidirectional
from keras.layers import GlobalMaxPool1D

!wget http://nlp.stanford.edu/data/glove.6B.zip
!apt-get -qq install unzip
!unzip glove.6B.zip
!pip install gensim

# number of most-frequent words to use
nb_words = 5000
n_classes = 1
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
# get_word_index retrieves a mapping word -> index
word_index = imdb.get_word_index()
# We make space for the three special tokens
word_index_c = dict((w, i+3) for (w, i) in word_index.items())
word_index_c['<PAD>'] = 0
word_index_c['<START>'] = 1
word_index_c['<UNK>'] = 2
# Instead of having dictionary word -> index we form
# the dictionary index -> word
index_word = dict((i, w) for (w, i) in word_index_c.items())
# Truncate sentences after this number of words
maxlen = 500

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")

glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

embedding_matrix = np.zeros((nb_words, 300))

for word, i in word_index_c.items():
    if word in glove_model:
      embedding_vector = glove_model[word]
      if embedding_vector is not None and i < nb_words:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector

## Model parameters:
# Dimensions of the embeddings
embedding_dim = 300

## LSTM dimensionality
lstm_units = 100

print('Build model...')
text_class_model = Sequential()

text_class_model.add(Embedding(nb_words,
                    embedding_dim,
                    input_length=maxlen,
                    weights=None,
                            trainable=True))

### Do not modify the layers below
text_class_model.add(Bidirectional(CuDNNLSTM(lstm_units, return_sequences = True)))
text_class_model.add(GlobalMaxPool1D())
text_class_model.add(Dense(1, activation='sigmoid'))
text_class_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(text_class_model.summary())

epochs = 10

randomhis = text_class_model.fit(x_train, y_train, batch_size=32, validation_split=0.2,epochs=epochs)

weights = text_class_model.layers[0].get_weights()[0]
query_word = 'action'
dist = ((weights - weights[word_index_c[query_word]])**2).sum(1).argsort()
top_k = 10
print('Most {:d} similar words to {:s}'.format(top_k, query_word))
for k in range(1, top_k+1):
  print("{:d}: {:s}".format(k, index_word[dist[k]]))




# code for printing comparison

plt.plot(randomhis.history['acc'],label= 'Random init Train-Accuracy')
plt.plot(randomhis.history['val_acc'],label= 'Random init Val-Accuracy')


plt.plot(embedhis.history['acc'],label= 'Embed-Matrix Train-Accuracy')
plt.plot(embedhis.history['val_acc'],label= 'Embed-Matrix Val-Accuracy')

plt.plot(embedntrainhis.history['acc'],label= 'Non Trainable Embed-Matrix Train-Accuracy')
plt.plot(embedntrainhis.history['val_acc'],label= 'Non Trainable Embed-Matrix Val-Accuracy')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(bbox_to_anchor=(1, 1))
plt.grid()
plt.show()

# plt.plot(vcg1his.history['loss'])
# plt.plot(vcg1his.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper right')
# plt.grid()
# plt.show()