import pandas
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
import math

windows = [1,3,6,12,18]
def create_dataset(dataset, window_size = 1):
    data_x, data_y = [], []
    for i in range(len(dataset) - window_size - 1):
        sample = dataset[i:(i + window_size), 0]
        data_x.append(sample)
        data_y.append(dataset[i + window_size, 0])
    return(np.array(data_x), np.array(data_y))

def get_predict_and_score(model, X, Y):
    # transform the prediction to the original scale.
    pred = normalizer.inverse_transform(model.predict(X))
    # transform also the label to the original scale for interpretability.
    orig_data = normalizer.inverse_transform([Y])
    # calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
data = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')

# convert pandas data frame in numpy array of float.
data_np = data.values.astype("float32")

# normalize data with min max normalization
normalizer = MinMaxScaler(feature_range = (0, 1))
dataset = normalizer.fit_transform(data_np)

# Using 70% of data for training, 30% for test.
TRAINING_PERC = 0.70
window_size = 1 #Use this variable to build the dataset with different number of inputs
batch_size = 32

train_size = int(len(dataset) * TRAINING_PERC)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print("Number of samples training set: " + str((len(train))))
# print("Number of samples test set: " + str((len(test))))
plt.figure(figsize = (15, 5))
ground = normalizer.inverse_transform(dataset)
plt.plot(ground[100:], label = "Ground Truth")

MSETRAIN = []
MSETEST = []

for window_size in windows:
  # Create test and training sets for regression with different window sizes.
  train_X, train_Y = create_dataset(train, window_size)
  test_X, test_Y = create_dataset(test, window_size)
  train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
  test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

  print("Shape of training inputs: " + str((train_X.shape)))
  print("Shape of training labels: " + str((train_Y.shape)))

  rnn = Sequential()    
  rnn.add(LSTM(16, input_shape = (window_size, 1)))
  rnn.add(Dense(1))
  rnn.compile(loss = "mean_squared_error",  optimizer = "adam", metrics = ['mse'])

  rnnhis=rnn.fit(train_X, train_Y, epochs=500, batch_size=batch_size, verbose = 0)

  mse_train, train_predict = get_predict_and_score(rnn, train_X, train_Y)
  mse_test, test_predict = get_predict_and_score(rnn, test_X, test_Y)

  print("Training data error: %.2f MSE" % mse_train)
  print("Test data error: %.2f MSE" % mse_test)

  MSETRAIN.append(mse_train)
  MSETEST.append(mse_test)

  # Training predictions.
  train_predictions = np.empty_like(dataset)
  train_predictions[:, :] = np.nan
  train_predictions[window_size:len(train_predict) + window_size, :] = train_predict

  # Test predictions.
  test_predictions = np.empty_like(dataset)
  test_predictions[:, :] = np.nan
  test_predictions[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict

  # Create the plot.
  
  
  # plt.plot(train_predictions, label = "Training predictions")
  plt.plot(test_predictions[100:], label = "Test predictions_"+str(window_size))
  # plt.plot(abs(ground[100:]-test_predictions[100:]),label="Error_"+str(window_size))
plt.xlabel("Months")
plt.ylabel("Error")
# plt.title("Comparison true vs. predicted in the training and testing set")
plt.title("Error vs. window size")
plt.legend(bbox_to_anchor=(1, 1))
plt.grid()
plt.show()

plt.plot(windows,MSETRAIN,label= 'mse_train')
plt.plot(windows,MSETEST,label= 'mse_test')
plt.xlabel("Window Size")
plt.ylabel("MSE")
plt.title("Mean Square Error vs. Window size")
plt.legend(bbox_to_anchor=(1, 1))
plt.grid()
plt.show()