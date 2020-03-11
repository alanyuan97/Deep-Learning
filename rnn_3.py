import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import CuDNNLSTM
from nltk.translate.bleu_score import sentence_bleu
import sys

download = True
temperatures = np.arange(0,1.1,0.1)
bleu_score = 0
n_eval = 500
results = []

if download:
  !git clone https://github.com/shekharkoirala/Game_of_Thrones
  data = open('./Game_of_Thrones/Data/final_data.txt', 'r').read()

characters = sorted(list(set(data)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}
x = []
y = []
length = len(data)
seq_length = 100

for i in range(0, length-seq_length, 2):
  sequence = data[i:i + seq_length]
  label = data[i + seq_length]
  x.append([char_to_n[char] for char in sequence])
  y.append(char_to_n[label])
n_samples = len(x)
print ('Total Samples:' , n_samples)

x_train = x[:int(n_samples*0.8)]
x_test = x[int(n_samples*0.8):]
y_train = y[:int(n_samples*0.8)]
y_test = y[int(n_samples*0.8):]

## Transform the list to a numpy array
x_train = np.reshape(x_train, (len(x_train), seq_length))
## Onehot encoding of labels
y_train = keras.utils.to_categorical(np.asarray(y_train))

x_test = np.reshape(x_test, (len(x_test), seq_length))
y_test = keras.utils.to_categorical(np.asarray(y_test))

# embedding_size = 300
# lstm_units = 256

# text_gen_model = Sequential()
# text_gen_model.add(Embedding(y_train.shape[1],
#                     embedding_size, input_length=seq_length))
# text_gen_model.add(CuDNNLSTM(lstm_units))
# text_gen_model.add(Dense(y_train.shape[1], activation='softmax'))

# text_gen_model.compile(loss='categorical_crossentropy', optimizer='adam')
# text_gen_model.summary()

if download:
  !wget https://imperialcollegelondon.box.com/shared/static/1ffasfm5bx691allukv4y8n0tglr5c06.h5 -O ./text_model.h5
text_gen_model = keras.models.load_model("./text_model.h5")

for temperature in temperatures:
  for _ in range(n_eval):
    start = np.random.randint(0, len(x_test)-seq_length-1)
    pattern = x_test[start].tolist()
    reference = x_test[start+seq_length].tolist()
    reference = ''.join([n_to_char[value] for value in reference]).split(' ')
    # generate characters
    output_sent = ''
    for i in range(100):
      x = np.reshape(pattern, (1, len(pattern)))
      prediction = text_gen_model.predict(x, verbose=0).astype(np.float64)
      prediction = np.log(prediction + 1e-7) / (temperature + 0.01)
      exp_preds = np.exp(prediction)
      prediction = exp_preds / np.sum(exp_preds)
      prediction = np.random.multinomial(1, prediction[0,:], 1)
      index = np.argmax(prediction)
      result = n_to_char[index]
      seq_in = [n_to_char[value] for value in pattern]
      output_sent += result
      pattern.append(index)
      pattern = pattern[1:]
    candidate = output_sent.replace('\n',' ').split(' ')
    bleu_score += sentence_bleu(reference, candidate)
    # print(bleu_score/n_eval)
  results.append(bleu_score/n_eval)
  print(f"Finished Temp {temperature}")

print(f"Temp: {temperatures}")
print(f"Score: {results}")
plt.plot(temperatures,results,label= 'BLUE score vs. Temperature')
plt.grid()
plt.legend()
plt.show()

# Temp: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
# Score: [0.33064787409782326, 0.7088449833692573, 1.1121576560614994, 1.5015591471042609, 1.8758053536668606, 2.246703281202218, 2.6216566240181742, 2.9617844281155663, 3.313229724077255, 3.6095638398928624, 3.9019044768169446]