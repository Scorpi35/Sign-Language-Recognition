from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

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

# Full Conenction
classifier.add(Dense(256, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation="softmax"))

# Compiling
classifier.compile(
    optimizer=optimizers.SGD(lr=0.01),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    "data/training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    "data/test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

model = classifier.fit_generator(
    training_set,
    steps_per_epoch=800,
    epochs=25,
    validation_data=test_set,
    validation_steps=6500
)






