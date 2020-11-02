from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(3, 64, 64), activation="relu", data_format='channels_first'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3,  3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3,  3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation='softmax'))

# Compiling CNN
classifier.compile(
              optimizer=optimizers.SGD(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generating image data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        directory=r"./data/training_set",
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical")


valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
        directory=r"./data/test_set",
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
)


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size


model = classifier.fit(train_generator,
               steps_per_epoch=STEP_SIZE_TRAIN,
               validation_data=valid_generator,
               validation_steps=STEP_SIZE_VALID,
               epochs=25)

classifier.save("Trained_model.h5")

# Visualize model history

plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







