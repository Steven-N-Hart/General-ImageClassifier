from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import re
import tensorflow as tf

tf.keras.backend.clear_session()
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, format_example_tf, update_status

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")
parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-L", "--loss-function",
                    dest='loss_function',
                    default='BinaryCrossentropy',
                    choices=['SparseCategoricalCrossentropy',
                             'CategoricalCrossentropy',
                             'BinaryCrossentropy'],
                    help="Loss functions from tf.keras")

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

parser.add_argument("-w", "--num-workers",
                    dest='NUM_WORKERS',
                    help="Number of workers to use for training",
                    default=1, type=int)

parser.add_argument("--use-multiprocessing",
                    help="Whether or not to use multiprocessing",
                    const=True, default=False, nargs='?',
                    type=bool)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="DEBUG",
                    help="Set the logging level")

parser.add_argument("-F", "--filetype",
                    dest="filetype",
                    choices=['tfrecords', 'images'],
                    default="images",
                    help="Set the logging level")

parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="Set the logging level")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="Set the logging level")

parser.add_argument("--train_num_layers",
                    dest="train_num_layers",
                    default="image/encoded",
                    help="Set the logging level")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label,
                        loss_function=args.loss_function)
logger.debug('Completed  training dataset Preprocess')

AUTOTUNE = 1000

# Get Validation data
# Update status to Testing for map function in the preprocess
update_status(False)
validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label,
                             loss_function=args.loss_function)
logger.debug('Completed test dataset Preprocess')

if validation_data.filetype != "tfrecords":
    v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
    v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    v_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(validation_data.labels, tf.int64))
    v_image_label_ds = tf.data.Dataset.zip((v_image_ds, v_label_ds))
else:
    v_path_ds = tf.data.TFRecordDataset(validation_data.files)
    v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    # min images variables should be update from number  of tfrecords to number of images
    num_image = 0
    for image, label in v_image_ds:
        num_image = num_image + 1
    validation_data.min_images = num_image
    v_image_label_ds = tf.data.Dataset.zip(v_image_ds)

validation_ds = v_image_label_ds.shuffle(buffer_size=validation_data.min_images).repeat()
validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Validation dataset')

out_dir = args.log_dir

checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

logger.debug('Training from {}'.format(latest))

# I now have generators for training and validation

###############################################################################
# Build the model
###############################################################################

logger.debug('Mirror initialized')
GPU = True  # TODO: Fix this hardcoded parameter
if GPU is True:
    # This must be fixed for multi-GPU
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if args.train_num_layers:
            m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes,
                         num_layers=int(args.train_num_layers))
        else:
            m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes)
        logger.debug('Model constructed')
        model = m.compile_model(args.optimizer, args.lr, args.loss_function)
    logger.debug('Model compiled')

    model.save_weights(checkpoint_path.format(epoch=0))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
    logger.debug('Loading initialized model')

    model.load_weights(latest)
    logger.debug('Completed loading initialized model')

    cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
    logger.debug('Model image saved')

    model.fit(train_ds,
              steps_per_epoch=training_steps,
              epochs=args.num_epochs,
              callbacks=cb.get_callbacks(),
              validation_data=validation_ds,
              validation_steps=validation_steps,
              class_weight=None,
              max_queue_size=1000,
              workers=args.NUM_WORKERS,
              use_multiprocessing=args.use_multiprocessing,
              shuffle=False, initial_epoch=ini_epoch
              )
    model.save(os.path.join(out_dir, 'my_model.h5'))
else:
    if args.train_num_layers:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes,
                     num_layers=int(args.train_num_layers))
    else:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes)
    logger.debug('Model constructed')

    model = m.compile_model(args.optimizer, args.lr, args.loss_function)
    logger.debug('Model compiled')

    model.save_weights(checkpoint_path.format(epoch=0))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
    logger.debug('Loading initialized model')

    model.load_weights(latest)
    logger.debug('Completed loading initialized model')

    cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
    logger.debug('Model image saved')

    model.fit(train_ds,
              steps_per_epoch=training_steps,
              epochs=args.num_epochs,
              callbacks=cb.get_callbacks(),
              validation_data=validation_ds,
              validation_steps=validation_steps,
              class_weight=None,
              max_queue_size=1000,
              workers=args.NUM_WORKERS,
              use_multiprocessing=args.use_multiprocessing,
              shuffle=False, initial_epoch=ini_epoch)
    model.save(os.path.join(out_dir, 'my_model.h5'))
