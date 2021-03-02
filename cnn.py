# code heavily based on https://www.tensorflow.org/tutorials/images/classification
# referenced/understanding from https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

### set up data ###
# assumes file structure
# biomes
# ..forest
# ....img
# ..icespike
# ....img
data_dir = pathlib.Path("biomes") #data_dir)

batch_size = 32
img_height = 128 #180
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

biome_names = train_ds.class_names

# load data from disk efficiently with cache, according to tf
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_biomes = 3

### create cnn ###
# DATA AUGMENTATION to reduce overfitting, generates additional training data by randomly transforming existing data
#                   exposes model to more aspects of data, helpful for small training datasets
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# CONV2D convolution layer to gather image features, ReLU activation function to increase non-linearity by setting negatives to zeros
# MAXPOOLING2D to reduce overfitting on features by pooling many output neurons into one neuron for the next layer based on max value in group of output neurons
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255), # make input values small for RGB, standardizing [0, 255] to [0, 1]
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # DROPOUT to reduce overfitting
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_biomes)
])

### compile and train ###
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

### save model ###
model.save("saved_models")

### evaluation ###
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig("plts/results.png")

### predict on new biome ###
new_biome_path = "shroom_test.jpg"

img = keras.preprocessing.image.load_img(
    new_biome_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(biome_names[np.argmax(score)], 100 * np.max(score))
)