import sys

import tensorflow as tf
from dataset import BATCH_SIZE, load_image_test, load_numpy
from model import Pix2Pix

train_data = sys.argv[1]
test_data = sys.argv[2]
train_results = sys.argv[3]
test_results = sys.argv[4]
logs = sys.argv[5]
checkpoint = sys.argv[6]

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("loading test tiffs", flush=True)
np_imgs_test = load_numpy(test_data)

print("loading test dataset", flush=True)
test_dataset = tf.data.Dataset.from_tensor_slices(np_imgs_test)
del np_imgs_test

print("mapping test dataset", flush=True)
test_dataset = test_dataset.map(load_image_test)

print("batching test dataset", flush=True)
test_dataset = test_dataset.batch(BATCH_SIZE)

print("creating model", flush=True)
model = Pix2Pix(logdir=logs, checkpoint_dir=checkpoint, train_dir=train_results, test_dir=test_results)
print("loading checkpoint", flush=True)
model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint))

print("Testing model", flush=True)

i = 1
for inp, tar in test_dataset.take(20):
    model.generate_images(inp, tar, test=True, step=i)
    i += 1