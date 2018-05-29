# In this model , the autonomous mode could run at 18mph.
import csv
from pprint import pprint
import cv2
import numpy as np
import sys
import pickle

import tensorflow as tf

print(tf.__version__)

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Flatten, Dense, Lambda
# from tensorflow.python.layers.convolutional import Convolution2D
# from tensorflow.python.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

reload_all_data = True

image_pickle_path = './image.piclke'
measurements_pickle_path = './measurements.pickle'
images = []
measurements = []


def load_data():
    """
    load all data
    """
    lines = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    # pprint(lines[:10])

    def fix_mesurement(mesurement, i):
        """
        fix left camera and right camera mesurement error
        """
        mesurement = mesurement
        if i is 1:  # left
            mesurement = mesurement + 0.30

        if i is 2:  # right
            mesurement = mesurement - 0.30
        return mesurement

    count = 0
    for line in lines:
        for i in range(3):
            image_path = '/nfs/project/car/data/IMG/' + line[i].split('/')[-1]
            image = cv2.imread(image_path)

            # origin image
            images.append(image)
            mesurement = float(line[3])
            measurements.append(fix_mesurement(mesurement, i))

            # flip image
            images.append(cv2.flip(image, 1))
            mesurement = float(line[3])
            measurements.append(-1.0 * fix_mesurement(mesurement, i))

        count += 1
        if count % 300 == 0:
            print("handle %d data. " % count)

    with open(image_pickle_path, 'wb') as f:
        pickle.dump(images, f)
    with open(measurements_pickle_path, 'wb') as f:
        pickle.dump(measurements, f)


if reload_all_data:
    load_data()
else:
    with open(image_pickle_path, 'rb') as f:
        images = pickle.load(f)
    with open(measurements_pickle_path, 'rb') as f:
        measurements = pickle.load(f)

print("begin to reshape data")
X_train = np.array(images)

print("train data size: ", X_train.shape)
y_train = np.array(measurements)

print("Train image size:", len(images))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(x=X_train, y=y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')

print("done")
