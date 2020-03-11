import numpy as np
import keras
np.random.seed(123)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from keras.datasets import mnist
from sklearn.decomposition import PCA

linear_trained = True
MSE=[]

def plot_recons_original(model,image, label):
  ## Function used to plot image x and reconstructed image \psi\phi(x)
  # Reshape (just in case) and predict using model
  if len(image.shape) == 1:
    image = image.reshape(1, -1)
  reconst_image = model.predict(image)
  # Evaluate MSE to also report it in the image
  mse = model.evaluate(image, image, verbose=0)
  # Create a figure with 1 row and 2 columns
  plt.subplots(1,2)
  # Select first image in the figure
  ax = plt.subplot(121)
  # Plot image x
  plt.imshow(image.reshape(28,28), cmap='gray')
  # This removes the ticks in the axis
  ax.set_xticks([])
  ax.set_yticks([])
  # Select second image in the figure
  ax = plt.subplot(122)
  # Plot reconstructed image
  plt.imshow(reconst_image.reshape(28,28), cmap='gray')
  # This removes the ticks in the axis
  ax.set_xticks([])
  ax.set_yticks([])
  # Set a title to the current axis (second figure)
  plt.title('MSE of {:.2f}'.format(mse[0]))
  plt.show()

def predict_representation(model, data, layer_name='representation'):
  ## We form a new model. Instead of doing \psi\phi(x), we only take \phi(x)
  ## To do so, we use the layer name
  intermediate_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer_name).output)
  representation = intermediate_layer_model.predict(data)
  representation = representation.reshape(representation.shape[0], -1)
  return representation

def plot_representation_label(representation, labels, plot3d=0):
  ## Function used to plot the representation vectors and assign different 
  ## colors to the different classes

  # First create the figure
  fig, ax = plt.subplots(figsize=(10,6))
  # In case representation dimension is 3, we can plot in a 3d projection too
  if plot3d:
    ax = fig.add_subplot(111, projection='3d')
    
  # Check number of labels to separate by colors
  n_labels = labels.max() + 1
  # Color map, and give different colors to every label
  cm = plt.get_cmap('gist_rainbow')
  ax.set_prop_cycle(color=[cm(1.*i/(n_labels)) for i in range(n_labels)])
  # Loop is to plot different color for each label
  for l in range(n_labels):
    # Only select indices for corresponding label
    ind = labels == l
    if plot3d:
      ax.scatter(representation[ind, 0], representation[ind, 1], 
                 representation[ind, 2], label=str(l))
    else:
      ax.scatter(representation[ind, 0], representation[ind, 1], label=str(l))
  ax.legend()
  plt.title('Features in the representation space with corresponding label')
  plt.show()
  return fig, ax

def cluster_plot_data(representation):
  from sklearn.cluster import KMeans
  # Set number of clusters to 10
  n_clusters = 10
  # Use KMeans
  c_pred = KMeans(n_clusters=n_clusters).fit_predict(representation)
  fig, ax = plt.subplots(figsize=(10,6))
  # Color map, and give different colors to every label
  cm = plt.get_cmap('gist_rainbow')
  ax.set_prop_cycle(color=[cm(1.*i/(n_clusters)) for i in range(n_clusters)])
  # Loop is to plot different color for each label
  for c in range(n_clusters):
    # Only select indices for corresponding label
    ind = c_pred == c
    ax.scatter(representation[ind, 0], representation[ind, 1], label=str(c))
  ax.legend()
  plt.title('Clustered features in the representation space')
  plt.show()

  correct = 0
  for i in range(10):
    indices_c_pred = c_pred == i
    classes = y_test[indices_c_pred]
    counts = np.bincount(classes)
    class_max = np.argmax(counts)
    correct += (classes == class_max).sum()
  
  print('Accuracy of {:.3f}'.format(correct/(1.0*y_test.shape[0])))
  return c_pred

  

num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

x_train= x_train.astype('float32')
x_test= x_test.astype('float32')

x_train /= 255.
x_test /= 255.

# Linear model skip if trained before
if not linear_trained:
  # Linear model
  linear_model = Sequential()
  linear_model.add(Dense(2, name='representation', input_shape=(28**2,)))
  linear_model.add(Dense(784))
  linear_model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mse'])
  print(linear_model.summary())
  epochs = 10
  validation_split = 0.1
  linear_history = linear_model.fit(x_train, x_train, batch_size=128,
            epochs=epochs, validation_split=validation_split)

# print('Test MSE obtained for linear model is {:.4f}'.format(linear_model.evaluate(x_test, x_test)[0]))


### Non-linear model
non_linear_model = Sequential()
non_linear_model.add(Dense(128, activation='relu', input_shape=(784,)))
non_linear_model.add(Dense(64, activation='relu'))
non_linear_model.add(Dense(4, name='representation'))
non_linear_model.add(Dense(64, activation='relu'))
non_linear_model.add(Dense(128, activation='relu'))
non_linear_model.add(Dense(784, activation='sigmoid'))
non_linear_model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
print(non_linear_model.summary())
epochs = 10
non_linear_history = non_linear_model.fit(x_train, x_train, batch_size=128,
          epochs=epochs, verbose=1)
mse = non_linear_model.evaluate(x_test, x_test)[0]
print('Test MSE obtained for Non linear_{} is {:.4f}'.format(i,mse))

for _ in range(1):
  ind = np.random.randint(x_test.shape[0] -  1)
  print("Start of Linear model ... ")
  plot_recons_original(linear_model,x_test[ind], y_test[ind])
  # Non linear model
  print("Start of NON-Linear model ...")
  plot_recons_original(non_linear_model,x_test[ind], y_test[ind])
  non_lin_representation = predict_representation(non_linear_model, x_test)

  pca = PCA(n_components=2)
  non_lin_pca = pca.fit(predict_representation(non_linear_model, x_train))
  representation_pca = non_lin_pca.transform(non_lin_representation)
  c_pred = cluster_plot_data(non_lin_representation)
  plot_representation_label(non_lin_representation, y_test,plot3d=0)