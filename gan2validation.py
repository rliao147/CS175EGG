import numpy as np 
from keras.models import load_model 
import matplotlib.pyplot as plt
import tensorflow as tf

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

def interpolate_points(p1, p2, n_steps=100):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = []
    for ratio in ratios:
        v = (1.0-ratio)*p1 + (ratio*p2)
        vectors.append(v)
    return np.asarray(vectors)

#load model
model = load_model('generator_model_56501.h5')

x_input = np.random.randn(100)
x_input = x_input.astype('float32')
# x_input = x_input/3
x_input2 =np.random.randn(100)
x_input2 = x_input2.astype('float32')
# x_input2 = x_input*3

steps_to_use = 200
interpolated = interpolate_points(x_input, x_input2, steps_to_use)

for noise in range(steps_to_use): 
    print("doing img " + str(noise))
    img = model.predict(np.asarray([interpolated[noise]]))
    img = (img+1) / 2.0
    fig, axs = plt.subplots(1, 1)
    axs.imshow(img[0])
    axs.axis('off')
    # plt.show()
    fig.savefig("spectrum/spectrum%d.png" % noise)
    plt.close() 

