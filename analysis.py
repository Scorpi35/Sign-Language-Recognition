from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape=(3, 64, 64), activation="relu", data_format='channels_first'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3,  3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3,  3), activation='relu'))
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

training_set = train_datagen.flow_from_directory(
        directory=r"./data/training_set",
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        class_mode='categorical')


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        directory=r"./data/test_set",
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        class_mode='categorical')






