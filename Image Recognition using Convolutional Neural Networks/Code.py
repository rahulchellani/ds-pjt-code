# -*- coding: utf-8 -*-
"""
IMAGE RECOGNITION USING CONVOLUTIONAL NEURAL NETWORKS
Rahul Chellani

PROBLEM STATEMENT:
TRAFFIC SIGN CLASSIFICATION. The dataset contains more than 50,000 images which are classified 
into 43 different classes.
Classes are as listed below:
( 0, b'Speed limit (20km/h)') ( 1, b'Speed limit (30km/h)')
( 2, b'Speed limit (50km/h)') ( 3, b'Speed limit (60km/h)')
( 4, b'Speed limit (70km/h)') ( 5, b'Speed limit (80km/h)')
( 6, b'End of speed limit (80km/h)') ( 7, b'Speed limit (100km/h)')
( 8, b'Speed limit (120km/h)') ( 9, b'No passing')
(10, b'No passing for vehicles over 3.5 metric tons')
(11, b'Right-of-way at the next intersection') (12, b'Priority road')
(13, b'Yield') (14, b'Stop') (15, b'No vehicles')
(16, b'Vehicles over 3.5 metric tons prohibited') (17, b'No entry')
(18, b'General caution') (19, b'Dangerous curve to the left')
(20, b'Dangerous curve to the right') (21, b'Double curve')
(22, b'Bumpy road') (23, b'Slippery road')
(24, b'Road narrows on the right') (25, b'Road work')
(26, b'Traffic signals') (27, b'Pedestrians') (28, b'Children crossing')
(29, b'Bicycles crossing') (30, b'Beware of ice/snow')
(31, b'Wild animals crossing')
(32, b'End of all speed and passing limits') (33, b'Turn right ahead')
(34, b'Turn left ahead') (35, b'Ahead only') (36, b'Go straight or right')
(37, b'Go straight or left') (38, b'Keep right') (39, b'Keep left')
(40, b'Roundabout mandatory') (41, b'End of no passing')
(42, b'End of no passing by vehicles over 3.5 metric tons')
    
OBJECTIVE:
Develop a CNN model which predicts and identifies the class of the new input images.
"""

##### IMPORTING DATA #####

import warnings
warnings.filterwarnings("ignore")

# Import libraries
import pickle
import seaborn as sns
#import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
#from keras.callbacks import TensorBoard
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import Data
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train.shape
y_train.shape


##### EDA #####

# Single Image Exploration
i = 1000
plt.imshow(X_train[i])
y_train[i]

i = 2000
plt.imshow(X_validation[i])
y_validation[i]

i = 3000
plt.imshow(X_test[i])
y_test[i]


##### DATA PREPROCESSING #####

# Shuffle the Dataset
X_train, y_train = shuffle(X_train, y_train)

X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray  = np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray  = np.sum(X_validation/3, axis=3, keepdims=True) 

X_train_gray.shape

X_train_gray_norm = (X_train_gray - 128)/128 
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128

i = 610
plt.imshow(X_train_gray[i].squeeze(), cmap='gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap='gray')

##### MODEL BUILDING #####

# Building the CNN
image_shape = X_train_gray[i].shape

cnn_model = Sequential()

cnn_model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=image_shape))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(rate=0.25))

cnn_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(rate=0.25))

cnn_model.add(Flatten())

cnn_model.add(Dense(units=120, activation='relu'))

cnn_model.add(Dense(units=84, activation='relu'))

cnn_model.add(Dense(units=43, activation = 'softmax'))


##### MODEL TRAINING #####

# Training the CNN
# Compiling the CNN
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

history = cnn_model.fit(X_train_gray_norm,
                        y_train,
                        batch_size=32,
                        epochs=25,
                        validation_data = (X_validation_gray_norm,y_validation))


##### MODEL EVALUATION #####

score = cnn_model.evaluate(X_test_gray_norm, y_test,verbose=0)
print('Test Accuracy : {:.4f}'.format(score[1]))

history.history.keys()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Get the predictions for the test data
predicted_classes = cnn_model.predict_classes(X_test_gray_norm)

# Get the indices to be plotted
y_true = y_test

cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25,25))
sns.heatmap(cm, annot=True)

# Visualize the Predicted vs Actual Classes of the Images
L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Prediction={}\n True={}".format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)