import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as nps
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import set_session

import matplotlib.pyplot as plt


#https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

BS=8
trdata = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        horizontal_flip=True, fill_mode="nearest")
traindata = trdata.flow_from_directory(directory="./train1",target_size=(300,300),batch_size=BS,class_mode="binary")
vldata = ImageDataGenerator()
valdata = vldata.flow_from_directory(directory="./val1", target_size=(300,300),batch_size=BS,class_mode="binary")


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

opt = opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001
)
model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.summary()


filenames = traindata.filenames
nb_samples = len(filenames)

model=tf.keras.models.load_model('./binary_classification_thread2.h5')

checkpoint = ModelCheckpoint("binary_classification_thread2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=nb_samples//BS,generator=traindata, validation_data= valdata, validation_steps=20,epochs=300,shuffle=True, callbacks=[checkpoint])



plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
