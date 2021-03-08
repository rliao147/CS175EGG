import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from PIL import Image

from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model 

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#load model
autoencoder = load_model('prevautoencoders/shrooms_ae_filter6_1.h5')

test_imgs_clean = []
test_imgs_masked = []

img_to_use = ImageDataGenerator().flow_from_directory("shrooms", color_mode='rgb', target_size = (128, 128), class_mode=None, batch_size=1)
img_to_use = img_to_use.next()
img_to_use = img_to_use.astype('float32')
img_to_use = img_to_use / 255
img_to_use = img_to_use[0]

current_mask_size = 25
mask = np.ones((128, 128, 3))
for i in range(20):
    min_x = 50 if 50+current_mask_size < 127 else 127-current_mask_size
    x = random.randint(min_x, 127-current_mask_size)
    y = random.randint(1, 127-current_mask_size)
    mask[x:x + current_mask_size,y:y + current_mask_size,:] = 0.0
    masked_img = np.multiply(img_to_use, mask)
    test_imgs_clean.append(img_to_use)
    test_imgs_masked.append(masked_img)

test_imgs_clean = np.array(test_imgs_clean)
test_imgs_masked = np.array(test_imgs_masked)

decoded_imgs = autoencoder.predict(test_imgs_masked)

n = 20
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(3, n, i + 1)
    plt.title("m")
    plt.imshow(tf.squeeze(test_imgs_masked[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(3, n, i + n + 1)
    plt.title("p")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

    # display reconstruction
    cx = plt.subplot(3, n, i + 2*n + 1)
    plt.title("a")
    plt.imshow(tf.squeeze(test_imgs_clean[i]))
    cx.get_xaxis().set_visible(False)
    cx.get_yaxis().set_visible(False)
plt.show() 