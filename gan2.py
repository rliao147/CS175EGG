from keras.layers import Input, Dense, Reshape, Flatten, Dropout 
# dense -- dense layers, all neurons of previous layer connected to a neuron in the next layer 
# reshape -- reshape the output of a layer to a specific shape
# flatten -- flatten the input by removing all of its dimneions except for one (basically turns matrix into an array)
# dropout -- essential to reduce overfitting. randomly sets the outgoing edges of hidden neurons to 0 
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
# batch normalization -- allows to train on a more stable distribution ofinputs, standardizes inputs to have a mean of 0 and std of 1 
# activation -- activation functions, transforms any input singal to an output singal for the next layer
from keras.layers.advanced_activations import LeakyReLU
# leakyrelu -- type of activation function, apparently very good in GAN 
from keras.models import Sequential, Model
# sequential -- type of model we build, means we build it layer by layer
from keras.optimizers import Adam
# adam -- optimizer function, apparently very good

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
# preprocess images

import matplotlib.pyplot as plt 

import sys 
import numpy as np 
from numpy.random import randn

class Network():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.d_model = self.define_discriminator()
        self.g_model = self.define_generator()
        self.gan_model = self.define_gan(self.g_model, self.d_model)
        self.dataset = self.load_real_samples()

    def define_discriminator(self):
        model = Sequential()
        
        model.add(Conv2D(64, (5,5), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        # model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(0.0002, 0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def define_generator(self):
        model = Sequential()
        model.add(Dense(200*4*4, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 200)))

        # model.add(Dense(512))

        # strides 2,2 upsamples by 2
        model.add(Conv2DTranspose(256, (6,6), strides=(2,2), padding='same'))
        # add regular conv layer??
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(256, (6,6), strides=(2,2), padding='same'))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(512, (2,2), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(256, (2,2), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, (2,2), activation='tanh', padding='same'))

        model.summary() 
        return model

    def define_gan(self, gen, desc):
        desc.trainable = False 

        model = Sequential()

        model.add(gen)
        model.add(desc)

        opt = Adam(0.0002, 0.5)

        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.summary()
        return model 

    def load_real_samples(self): 
        # should later change so that we only load one batch into directory
        X_train = ImageDataGenerator().flow_from_directory('train', color_mode='rgb', target_size = (self.img_rows, self.img_cols), class_mode=None, batch_size=1858)
        X_train = X_train.next()
        X = X_train.astype('float32')
        X = (X - 127.5) / 127.5
        # print(X)
        return X 

    def generate_real_samples(self, batch_size):
        # X_train = ImageDataGenerator().flow_from_directory('train', color_mode='rgb', target_size = (self.img_rows, self.img_cols), class_mode=None, batch_size=batch_size)
        # X_train = X_train.next()
        # X = X_train.astype('float32')
        # X = (X - 127.5) / 127.5
        idx = np.random.randint(0, self.dataset.shape[0], batch_size)
        imgs = self.dataset[idx]
        return imgs
    
    def generate_latent_points(self, batch_size):
        x = randn(self.latent_dim * batch_size)
        x = x.reshape(batch_size, self.latent_dim)
        return x 

    def generate_fake_samples(self, batch_size):
        x_input = self.generate_latent_points(batch_size)
        X = self.g_model.predict(x_input)
        return X 
    
    def sample_images(self, epoch):
        r, c = 5, 5
        gen_imgs = self.generate_fake_samples(r*c)
        gen_imgs = (gen_imgs + 1) / 2.0
        # gen_imgs = gen_imgs*255
        # print(gen_imgs.shape)
        # print(gen_imgs[0])

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range (r):
            for j in range (c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig("images/%d.png" % epoch)
        plt.close() 
        filename = 'generator_model_%d.h5' % (epoch)
        self.g_model.save(filename)

        # mask parts of images and generate 

    def train(self, epochs, batch_size, sample_interval):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            imgs = self.generate_real_samples(batch_size)
            d_loss_real = self.d_model.train_on_batch(imgs, valid)

            gen_imgs = self.generate_fake_samples(batch_size)
            d_loss_fake = self.d_model.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            X_gan = self.generate_latent_points(batch_size)
            g_loss = self.gan_model.train_on_batch(X_gan, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

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


    # try:
    #     # Disable all GPUS
    #     tf.config.set_visible_devices([], 'GPU')
    #     visible_devices = tf.config.get_visible_devices()
    #     for device in visible_devices:
    #         assert device.device_type != 'GPU'
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass

    #     gpus = tf.config.experimental.list_physical_devices('GPU')

    #     config = tf.compat.v1.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.compat.v1.Session(config=config)

    gan = Network()
    gan.train(epochs=100000, batch_size = 32, sample_interval = 500)

    



