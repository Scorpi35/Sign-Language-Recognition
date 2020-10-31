from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers

# CNN
classifier = Sequential()

# Convolution layer

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolution layer
classifier.add(Convolution2D(64, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

