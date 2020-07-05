
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


batch_size = 8

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './train1',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './val1',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')


#model=tf.keras.models.load_model('./binary_classification_thread2.h5')

checkpoint = ModelCheckpoint("binary_classification_model4.h5", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=3)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#hist = model.fit_generator(steps_per_epoch=nb_samples//BS,generator=traindata, validation_data= valdata, validation_steps=20,epochs=300,shuffle=True, callbacks=[checkpoint])


num_train=len(train_generator.filenames)

num_val=len(validation_generator.filenames)


hist=model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,shuffle=True, callbacks=[checkpoint])

plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()