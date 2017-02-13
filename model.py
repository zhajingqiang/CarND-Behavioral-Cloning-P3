from keras.models import Sequential
from keras.layers.core import Flatten, Activation, Dense, Lambda, Dropout
from keras.activations import softmax, relu
from keras.models import model_from_json
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers import Cropping2D
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
# import tensorflow as tf 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/'+batch_sample[0]
                center_image = Image.open(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#read data 
in_file = 'data/driving_log.csv'
full_data = pd.read_csv(in_file)
image_names = full_data['center']
steer_data = full_data['steering']
image_first = np.array(Image.open("data/" + image_names[0]))
train_samples, validation_samples = train_test_split(full_data, test_size=0.2)
train_generator = generator(train_samples)
valid_generator = generator(validation_samples)

# plt.imshow(image_first)
# plt.show()
# print(image_first.shape)

#use left and right cameras
# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         steering_center = float(row[3])

#         # create adjusted steering measurements for the side camera images
#         correction = 0.2 # this is a parameter to tune
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction

#         # read in images from center, left and right cameras
#         directory = "..." # fill in the path to your training IMG directory
#         img_center = process_image(np.asarray(Image.open(path + row[0])))
#         img_left = process_image(np.asarray(Image.open(path + row[1])))
#         img_right = process_image(np.asarray(Image.open(path + row[2])))

#         # add images and angles to data set
#         car_images.extend(img_center, img_left, img_right)
#         steering_angles.extend(steering_center, steering_left, steering_righ

# X_train = np.zeros((len(image_names)*2,image_first.shape[0],image_first.shape[1],image_first.shape[2]))
# y_train = np.zeros((len(image_names)*2,1))
# for i in range(len(image_names)): 
# 	X_train[i*2] = np.array(Image.open("data/" + image_names[i]))
# 	X_train[i*2+1] = np.array(np.fliplr(Image.open("data/" + image_names[i])))
# 	y_train[i*2] = steer_data[i]
# 	y_train[i*2+1] = -steer_data[i]

# build architecture
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape = image_first.shape))
# Crop image
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(240))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
# history = model.fit(X_train, y_train,  batch_size=32, nb_epoch=10, validation_split=0.2)
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save_weights('./model.h5')
json_string = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(json_string)