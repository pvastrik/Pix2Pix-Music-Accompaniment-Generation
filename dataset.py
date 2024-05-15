import numpy as np
import tensorflow as tf
import os


def load_numpy(path):
    num = len(os.listdir(path))
    imgs = np.ndarray((num, 512, 256, 1))
    for i, filename in enumerate(os.listdir(path)):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and f.endswith(".npz"):
            with np.load(f, allow_pickle=True) as data:
                imgs[i] = np.expand_dims(data['arr_0'], 2)
    return imgs

def load(image):

    h = tf.shape(image)[0]
    h = h // 2
    input_image = image[h:, :, :]
    real_image = image[:h, :, :]
    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


BUFFER_SIZE = 1000
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 40) + 1
    real_image = (real_image / 40 ) + 1

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
