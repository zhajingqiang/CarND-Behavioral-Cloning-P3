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
from sklearn.utils import shuffle
from PIL import Image
# import tensorflow as tf 
def generator(image_paths, steering, batch_size=32):
    num_samples = len(image_paths)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(image_paths,steering)
        for offset in range(0, num_samples, batch_size):
            batch_images = image_paths[offset:offset+batch_size]
            batch_angles = steering[offset:offset+batch_size]
            images = []
            angles = []
            for i in range(len(batch_images)):
                name = 'data/'+batch_images[i]
                image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                angle = float(batch_angles[i])
                images.append(image)
                images.append(np.fliplr(image))
                angles.append(angle)
                angles.append(-angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#read data 
in_file = 'data/driving_log.csv'
full_data = pd.read_csv(in_file)
# data shift setting
image_paths = []
steering_angle = []
angle_shift = 0.22
image_left = full_data['left']
angle_left = full_data['steering'] + angle_shift
image_center = full_data['center']
angle_center = full_data['steering']
image_right = full_data['right']
angle_right = full_data['steering'] - angle_shift

for i in range(len(image_center)):
    image_paths.append(image_center[i])
    steering_angle.append(angle_center[i])
    image_paths.append(image_left[i].strip())
    steering_angle.append(angle_left[i])
    image_paths.append(image_right[i].strip())
    steering_angle.append(angle_right[i])
image_paths_train, image_paths_validation, steering_train,steering_validation = train_test_split(image_paths, steering_angle, test_size=0.2)

train_generator = generator(image_paths_train, steering_train)
validation_generator = generator(image_paths_validation, steering_validation)

image_first = np.array(Image.open("data/" + image_paths_train[0]))
plt.imshow(image_first)
plt.show()
# print(image_first.shape)

# for i in range(len(train_samples)):
#     image = train_samples['center']
#     print(image[int(i)])
    # name = 'data/'+train_samples['center'][i]
    # center_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
    # center_angle = float(train_samples['steering'][i])
    # images.append(center_image)
    # angles.append(center_angle)
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
history = model.fit_generator(train_generator, samples_per_epoch=len(image_paths_train), validation_data=validation_generator, nb_val_samples=len(image_paths_validation), nb_epoch=10)
model.save_weights('./model.h5')
json_string = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(json_string)