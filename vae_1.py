import numpy as np
import keras
np.random.seed(123)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Layer, Input, Lambda 
from keras.layers import Multiply, Add, BatchNormalization, Reshape
from keras.layers import UpSampling2D, Convolution2D, LeakyReLU, Flatten, ReLU


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from keras.datasets import mnist
from keras import backend as K
from scipy.stats import norm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
import matplotlib.image as mpimg
import sys

from tqdm import tqdm_notebook
from IPython import display
%matplotlib inline

from keras import initializers
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D

!wget https://imperialcollegelondon.box.com/shared/static/5cc14wf0s4qwj65lec5852jlmxfy32m9.h5 -O inception_score_mnist.h5
inception_score_model = keras.models.load_model('./inception_score_mnist.h5')

original_dim = 784
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.   

def nll(y_true, y_pred):
  """ Negative log likelihood (Bernoulli). """

  # keras.losses.binary_crossentropy gives the mean
  # over the last axis. we require the sum
  return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class KLDivergenceLayer(Layer):

  """ Identity transform layer that adds KL divergence
  to the final model loss.
  """

  def __init__(self, *args, **kwargs):
    self.is_placeholder = True
    super(KLDivergenceLayer, self).__init__(*args, **kwargs)

  def call(self, inputs):
    mu, log_var = inputs
    kl_batch = - .5 * K.sum(1 + log_var -
                            K.square(mu) -
                            K.exp(log_var), axis=-1)
    self.add_loss(K.mean(kl_batch), inputs=inputs)

    return inputs
def inception_score(x, resizer=None, batch_size=32, denorm_im=1):
    r = None
    n_batch = (x.shape[0]+batch_size-1) // batch_size
    for j in range(n_batch):
        x_batch = x[j*batch_size:(j+1)*batch_size, :, :, :]
        if denorm_im:
          x_batch = (x_batch + 1)/2
        r_batch = inception_score_model.predict(x_batch) # r has the probabilities for all classes
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    p_y = np.mean(r, axis=0) # p(y)
    e = r*np.log(r/p_y) # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1) # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e) # Inception score


def image_inception_score(generator, n_ex=10000, dim_random=10, input_noise=None, denorm_im=1):
    if input_noise is None:
      input_noise = np.random.normal(0,1,size=[n_ex,dim_random])
    x_pred = generator.predict(input_noise)
    if len(x_pred.shape)==2:
      x_pred = x_pred.reshape(n_ex, 28, 28, 1)
    return inception_score(x_pred, denorm_im=denorm_im)


MAE = []
Iscore = []

## Encoder
intermediate_dim = 256
# latent_dim = 2

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
h = Dense(intermediate_dim, activation='relu')(h)

for latent_dim in range(2,21):
  ## We recover here \mu and \sigma
  ## For stability purposes, we assume that it outputs \log(\sigma)
  z_mu = Dense(latent_dim)(h)
  z_log_var = Dense(latent_dim)(h)

  ## This layer adds the KL loss we defined before to the model
  z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

  ##### Reparametrisation trick

  ## Log_var to sigma
  z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

  ## Sample using normal distribution
  eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)))

  ## Multiply by sigma
  z_eps = Multiply()([z_sigma, eps])

  ## Add mu
  z = Add()([z_mu, z_eps])

  decoder = Sequential([
      Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
      Dense(intermediate_dim, activation='relu'),
      Dense(original_dim, activation='sigmoid')
  ])
    
  x_pred = decoder(z)
  vae = Model(inputs=[x, eps], outputs=x_pred)
  vae.compile(optimizer='rmsprop', loss=nll, metrics = ['mae'])

  epochs = 20
  batch_size = 50
  vae.fit(x_train,
          x_train,
          shuffle=True,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test, x_test))

  _, mae = vae.evaluate(x_test, x_test)

  print('VAE Model MAE: ', mae)

  inceptionres = image_inception_score(decoder, dim_random=latent_dim, denorm_im=0)

  print('VAE Model Inception: ', inceptionres)
  MAE.append(mae)
  Iscore.append(inceptionres)
  
print('MAE:',MAE)
print('Inception Score:',Iscore)