import sys

import tensorflow as tf
from dataset import load_image_train, BUFFER_SIZE, BATCH_SIZE, load_image_test, load_numpy
from model import Pix2Pix

print("creating strategy", flush=True)

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(gpus_per_node=2,tasks_per_node=2,port_base=1234),
    communication_options=communication_options
)
train_data = sys.argv[1]
test_data = sys.argv[2]
train_results = sys.argv[3]
test_results = sys.argv[4]
logs = sys.argv[5]
checkpoint = sys.argv[6]

print("loading train tiffs", flush=True)
np_imgs_train = load_numpy(train_data)

print("loading train dataset", flush=True)
train_dataset = tf.data.Dataset.from_tensor_slices(np_imgs_train)
del np_imgs_train

print("mapping train dataset", flush=True)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)

print("shuffling train dataset", flush=True)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)

print("batching train dataset", flush=True)
train_dataset = train_dataset.batch(BATCH_SIZE)

print("loading test tiffs", flush=True)
np_imgs_test = load_numpy(test_data)

print("loading test dataset", flush=True)
test_dataset = tf.data.Dataset.from_tensor_slices(np_imgs_test)
del np_imgs_test

print("mapping test dataset", flush=True)
test_dataset = test_dataset.map(load_image_test)

print("batching test dataset", flush=True)
test_dataset = test_dataset.batch(BATCH_SIZE)

print('Number of devices: {}'.format(strategy.num_replicas_in_sync), flush=True)
with strategy.scope():
    print("creating model", flush=True)
    model = Pix2Pix(logdir=logs, checkpoint_dir=checkpoint, train_dir=train_results, test_dir=test_results)

print("Training model", flush=True)
model.fit(train_dataset, test_dataset, 1, 30000)

print("Testing model", flush=True)

i = 1
for inp, tar in test_dataset.take(10):
    model.generate_images(inp, tar, test=True, step=i)
    i += 1
