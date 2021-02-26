import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from PIL import Image

# Gather training and test data 

train_clear = []
train_mask = []
test_clear = []
test_mask = []

for i in range(1, 5101):
    img = Image.open("./img/" + str(i) + ".jpg")
    pixels = np.array(img)
    mask_img = Image.open("./imgmasked/" + str(i) + "masked.jpg")
    mask_pixels = np.array(mask_img)
    if i <= 5000:
        train_clear.append(pixels)
        train_mask.append(mask_pixels)
    else:
        test_clear.append(pixels)
        test_mask.append(mask_pixels)
    img.close()
    mask_img.close()

# Convert all numpy arrays to floating numbers range: [0...1]
train_clear = np.array(train_clear)
x_train_clear = train_clear.astype('float32') / 255.
train_mask = np.array(train_mask)
x_train_mask = train_mask.astype('float32') / 255.
test_clear = np.array(test_clear)
x_test_clear = test_clear.astype('float32') / 255.
test_mask = np.array(test_mask)
x_test_mask = test_mask.astype('float32') / 255.

# Verify that the shape is (height, width, 3 (for RGB)) for all np arrays
print(x_train_clear.shape, x_test_clear.shape, x_train_mask.shape, x_test_mask.shape)

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256, 256, 3)), 
      layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
      # add this layer if your CPU can handle it
      #layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      # add this layer if your CPU can handle it
      #layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train_mask, x_train_clear,
# edit number of epochs here
                epochs=25,
                shuffle=True,
                validation_data=(x_test_mask, x_test_clear))
autoencoder.encoder.summary()
autoencoder.decoder.summary()


encoded_imgs = autoencoder.encoder(x_test_mask).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_mask[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()