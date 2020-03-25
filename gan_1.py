def plot_loss(losses):
    plt.figure()
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.savefig('./loss.png')
    plt.close()
def plot_gen(mnist=1, n_ex=16, dim=(4,4), figsize=(10,10)):
    noise = np.random.normal(0,1,size=[n_ex,randomDim])
    generated_images = generator.predict(noise)
    
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        if mnist:
          img = generated_images[i,:,:,0]
          plt.imshow(img, cmap='gray')
        else:
          img = generated_images[i,:,:,:]
          plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images.png')
    plt.close()

def train_epoch(gan, generator, discriminator, plt_frq=25, BATCH_SIZE=32, mnist=1):  
    vector_ind = np.random.permutation(x_train.shape[0])
    nb_epoch = int(x_train.shape[0]/BATCH_SIZE)
    pbar = tqdm_notebook(range(nb_epoch))
    for e in range(nb_epoch):  
        ind = vector_ind[e*BATCH_SIZE:(e+1)*BATCH_SIZE]
        # Make generative images
        image_batch = x_train[ind,:,:,:]    
        noise_gen = np.random.normal(0,1,size=[BATCH_SIZE,randomDim])
        generated_images = generator.predict(noise_gen)
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE])
        y[0:BATCH_SIZE] = 1
        y[BATCH_SIZE:] = 0

        #make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.normal(0,1,size=[BATCH_SIZE,randomDim])
        y2 = np.zeros([BATCH_SIZE])
        y2[:] = 1

        #make_trainable(discriminator,False)
        g_loss = gan.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)

        # Updates plots. This is a little bit of a mess due to how the notebook
        # handles the outputs
        if e % plt_frq==plt_frq-1:
          plot_loss(losses)
          plot_gen(mnist)
          fig, ax = plt.subplots(2,1, figsize=(20,10) )
          img=mpimg.imread('loss.png')
          ax[0].imshow(img)
          ax[0].axis('off')
          img=mpimg.imread('images.png')
          ax[1].imshow(img)
          ax[1].axis('off')
          plt.tight_layout()
          display.clear_output(wait=True)
          pbar.update(plt_frq)
          display.display(pbar)
          display.display(fig)
          plt.close()

# Build Generative model
# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
randomDim = 10
# Generator
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)
generator.summary()

# Build Discriminative model ...
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
discriminator.summary()

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)
gan.summary()

# set up loss storage vector
losses = {"d":[], "g":[]}
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 127.5 - 1
x_test = x_test.reshape(-1, 28, 28, 1) / 127.5 - 1 
n_epoch = 10
for i in range(n_epoch):
  train_epoch(gan, generator, discriminator, plt_frq=200,BATCH_SIZE=32)

res = image_inception_score(generator, dim_random=randomDim, denorm_im=1)

print("IS: ", res)