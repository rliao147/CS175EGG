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

# Gather training and test data

image_size_used = 128

def get_images(directory):
  imgs = ImageDataGenerator().flow_from_directory(directory, color_mode='rgb', target_size = (image_size_used, image_size_used), class_mode=None, batch_size=(28733))
  imgs = imgs.next()
  imgs = imgs.astype('float32')
  ret = imgs / 255
  ret = np.array(ret)
  return ret

if __name__ == '__main__':
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

    imgs = get_images("trainae")
    
    
    train_clear = []
    train_mask = []
    test_clear = []
    test_mask = []

    counter = 0
    for img in imgs: 
        mask = np.ones((image_size_used, image_size_used, 3))
        mask_size_x = random.randint(16, 64)
        mask_size_y = random.randint(16, 64)
        x = random.randint(1, image_size_used-1-mask_size_x)
        y = random.randint(1, image_size_used-1-mask_size_y-(image_size_used//4))
        mask[x:x + mask_size_x,y:y + mask_size_y,:] = 0.0
        masked_img = np.multiply(img, mask)
        if counter < 28713:
            train_clear.append(img)
            train_mask.append(masked_img)
        else: 
            test_clear.append(img)
            test_mask.append(masked_img)
        counter+=1

    train_clear = np.array(train_clear)
    train_mask = np.array(train_mask)
    test_clear = np.array(test_clear)
    test_mask = np.array(test_mask)

    # Verify that the shape is (height, width, 3 (for RGB)) for all np arrays
    print(train_clear.shape, test_clear.shape, train_mask.shape, test_mask.shape)

    input_img = keras.Input(shape=(image_size_used, image_size_used, 3))    

    x = layers.Conv2D(300, (7, 7), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(300, (7, 7), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # At this point the representation is (7, 7, 32)

    x = layers.Conv2D(300, (7, 7), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(300, (7, 7), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    print(autoencoder.summary())

    history = autoencoder.fit(train_mask, train_clear,
                    epochs=16,
                    batch_size=32,
                    shuffle=True)
    #                 #,
    #                 #validation_data=(test_mask, test_clear))

    # autoencoder.fit(train_mask, train_clear,
    #                 epochs=150,
    #                 batch_size=32,
    #                 shuffle=True)

    filename = 'autoencoder.h5' 
    autoencoder.save(filename)


    decoded_imgs = autoencoder.predict(test_mask)

    n = 20
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(3, n, i + 1)
        plt.title("m")
        plt.imshow(tf.squeeze(test_mask[i]))
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
        plt.imshow(tf.squeeze(test_clear[i]))
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)
    plt.show()

    loss_to_plot = history.history['loss']
    # val_loss_to_plot = history.history['val_loss']
    
    epoch_range = range(1, len(loss_to_plot)+1)
    losses = plt.subplot(4, 2, 1)
    plt.plot(epoch_range, loss_to_plot, label='Training Loss')
    # plt.plot(epoch_range, val_loss_to_plot, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.show()

    
