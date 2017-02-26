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
from numpy import random
# import tensorflow as tf 


# reference "An augmentation based deep neural network approach to learn human driving behavior" by Vivek Yadav
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    # Translation
    rows, cols, _ = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def generator(images, steering, batch_size=32):
    num_samples = len(images)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(images,steering)
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset+batch_size]
            batch_angles = steering[offset:offset+batch_size]
            X_train = np.array(batch_images)
            y_train = np.array(batch_angles)
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

images_augmentation = []
steer_augmentation = []
image_validation = []
for i in range(len(image_paths_train)):
    name = 'data/'+image_paths_train[i]
    image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
    for j in range(2):
        # augment data
        flag = np.random.randint(2)
        if flag ==0:
            image_shift,steer_shift = trans_image(image,steering_train[i],100)
            image_processed = add_random_shadow(augment_brightness_camera_images(image_shift))
        else:
            image_shift,steer_shift = trans_image(np.fliplr(image),-steering_train[i],100)
            image_processed = add_random_shadow(augment_brightness_camera_images(image_shift))
        images_augmentation.append(image_processed)
        steer_augmentation.append(steer_shift)

for i in range(len(image_paths_validation)):
    name = 'data/'+image_paths_validation[i]
    image_valid = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
    image_validation.append(image_valid)

train_generator = generator(images_augmentation, steer_augmentation)
validation_generator = generator(image_validation, steering_validation)

image_first = np.array(Image.open("data/" + image_paths_validation[100]))
# plt.imshow(image_first)
# plt.show()
# plt.imshow(np.fliplr(image_first))
# plt.show()
# print(steering_train[100])
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


# build architecture
model = Sequential()
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape = image_first.shape))
model.add(Lambda(lambda x: x/127.5 - 1.0))
# Crop image
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
# history = model.fit(X_train, y_train,  batch_size=32, nb_epoch=10, validation_split=0.2)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(images_augmentation), validation_data=validation_generator, nb_val_samples=len(image_validation), nb_epoch=6, verbose = 1)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save_weights('./model.h5')
json_string = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(json_string)