# Dataset Link [https://www.kaggle.com/competitions/cifar-10]


import py7zr

archive = py7zr.SevenZipFile('/kaggle/input/cifar-10/train.7z', mode='r')
archive.extractall()     #archive.extractall(path='/content/Training Data')
archive.close()

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

filenames = os.listdir('/kaggle/working/train')
print(filenames[0:5])
print(filenames[-5:])


# Labels Processing
df = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
df.head()
df[df['id'] == 7796]
df['label'].value_counts()
df['label'].value_counts()
labels_dictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
labels = [labels_dictionary[i] for i in df['label']]
print(labels[0:5])
print(labels[-5:])
#import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/kaggle/working/train/7796.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# displaying sample image
from matplotlib import pyplot as plt

img = cv2.imread('/kaggle/working/train/45888.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
df[df['id'] == 45888]
df.head()
id_list = list(df['id'])
print(id_list[0:5])
print(id_list[-5:])

#  Image Processing
#convert images to numpy arrays

train_data_folder = '/kaggle/working/train/'

data = []

for id in id_list:

  image = Image.open(train_data_folder + str(id) + '.png')
  image = np.array(image)
  data.append(image)
type(data),len(data),type(data[0]),data[0].shape
data[0] 
# convert image list and label list to numpy arrays

x = np.array(data)
y = np.array(labels)
type(x)


# Training
# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)
# scaling the data

x_train_scaled = x_train/255
x_test_scaled = x_test/255
Building the Neural Network
import tensorflow as tf
from tensorflow import keras
num_of_classes = 10

# setting up the layers of Neural Network

model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])
# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
# training the neural network
model.fit(x_train_scaled, y_train, validation_split=0.1, epochs=10)
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print('Test Accuracy =', accuracy) # accuracy is low so we'll use ResNet50

# ResNet50 

from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers
convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
num_of_classes = 10

model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(convolutional_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_of_classes, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train_scaled, y_train, validation_split=0.1, epochs=10)
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print('Test Accuracy =', accuracy)
h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
