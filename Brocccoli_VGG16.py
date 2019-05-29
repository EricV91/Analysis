# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:44:31 2019

@author: VanBoven
"""

from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow
import time, os, shutil
import random

import os, shutil
import matplotlib.pyplot as plt

import sys
from PIL import Image
sys.modules['Image'] = Image 

from PIL import Image
print(Image.__file__)

import Image
print(Image.__file__)

import os

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = r'E:\400 Data analysis\410 Plant count\Training_data'
# The directory where we will
# store our smaller dataset
base_dir = r'E:\400 Data analysis\410 Plant count\Training_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


from tensorflow.keras.applications.vgg16 import VGG16
# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights='imagenet')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(50, 50, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Dense(4, activation='softmax'))

conv_base.trainable = False

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(50, 50),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(50, 50),
    batch_size=20,
    class_mode='categorical')


model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

model.save(r'C:\Users\VanBoven\Documents\GitHub\DataAnalysis/Broccoli_VGG16_fine_tuned.h5')



