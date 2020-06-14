#Dogs and Cats DataSet
#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing CNN
classifier = Sequential()

#1Convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))

#2Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#adding 2nd convolution layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#3Flattening
classifier.add(Flatten())

#4Full_Connection
classifier.add(Dense(output_dim=128,activation = 'relu'))
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))

#Compiling CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = train_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data =test_set,
                         nb_val_samples=2000)

#Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_image.png',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] ==1:
    prediction = 'Prediction Result : Affected'
else:
    prediction = 'Prediction Result : Unaffected'
