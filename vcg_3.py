from keras.applications.vgg16 import VGG16,VCG19
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense,Lambda
import tensorflow as ktf
ktf.logging.set_verbosity(ktf.logging.ERROR)
from time import time

#Load the VGG
model16 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model19 = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# Freeze all the layers
for layer in model16.layers[:]:
    layer.trainable = False

for layer in model19.layers[:]:
    layer.trainable = False

newInput = Input(batch_shape=(None, 64, 64, 3))
resizedImg = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(newInput)
newOutputs16 = model16(resizedImg)
newOutputs19 = model19(resizedImg)
model16 = Model(newInput, newOutputs16)
model19 = Model(newInput, newOutputs19)
# Add Dense layer as in VGG16
output16 = model16.output
output16 = Dense(units=200, activation='softmax')(output16)
model16 = Model(model16.input, output16)
model16.summary()

output19 = model19.output
output19 = Dense(units=200, activation='softmax')(output19)
model19 = Model(model19.input, output19)
model19.summary()

early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=4)
model16.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
time_16 = time()
vcg16his = model.fit(train_data, train_labels, epochs=20, batch_size=128,  validation_split=0.2,callbacks=[early_stopping])
16_exetime = time()-time_16

model19.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
time_19 = time()
vcg19his = model.fit(train_data, train_labels, epochs=20, batch_size=128,  validation_split=0.2,callbacks=[early_stopping])
19_exetime = time()-time_19


plt.plot(vcg16his.history['categorical_accuracy'])
plt.plot(vcg16his.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.show()

plt.plot(vcg19his.history['loss'])
plt.plot(vcg19his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid()
plt.show()

print(f"Execution time VCG16: {16_exetime}s \n Execution time VCG19: {19_exetime}s")