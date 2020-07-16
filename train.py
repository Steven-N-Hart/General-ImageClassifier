from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import re
import tensorflow as tf
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, update_status


###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_dir_train",
                    dest='image_dir_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")

parser.add_argument("-m", "--model-name",
                    dest='model_name',
                    default='VGG16',
                    choices=['DenseNet121',
                             'DenseNet169',
                             'DenseNet201',
                             'InceptionResNetV2',
                             'InceptionV3',
                             'MobileNet',
                             'MobileNetV2',
                             'NASNetLarge',
                             'NASNetMobile',
                             'ResNet50',
                             'ResNet152',
                             'VGG16',
                             'VGG19',
                             'Xception'],
                    help="Models available from tf.keras")

parser.add_argument("-o", "--optimizer-name",
                    dest='optimizer',
                    default='Adam',
                    choices=['Adadelta',
                             'Adagrad',
                             'Adam',
                             'Adamax',
                             'Ftrl',
                             'Nadam',
                             'RMSprop',
                             'SGD'],
                    help="Optimizers from tf.keras")

parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-r", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.0001, type=float)

parser.add_argument("-L", "--loss-function",
                    dest='loss_function',
                    default='BinaryCrossentropy',
                    choices=['SparseCategoricalCrossentropy',
                             'CategoricalCrossentropy',
                             'BinaryCrossentropy','Hinge'],
                    help="Loss functions from tf.keras")

parser.add_argument("-e", "--num-epochs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    type=int,
                    help="Number of batches to use for training",
                    default=1)

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
                    help="What type of input are you using?")

parser.add_argument("-D", "--drop_out",
                    dest="reg_drop_out_per",
                    default=None, type=float,
                    help="Regularization drop out percent 0-1")

parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="What is the name of the key in the tf record that contains the IMAGE you want to use?")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="What is the name of the key in the tf record that contains the LABEL you want to use?")

parser.add_argument("-N", "--train_num_layers",
                    dest="train_num_layers",
                    type=int,
                    default=False,
                    help="How many layers from the bottom should be trainable")

parser.add_argument("-P", "--prev_checkpoint",
                    dest="prefix",
                    default=None,
                    help="Add a prefix to the name of this run")


args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Begin priming the data generation pipeline
###############################################################################
print(tf.__version__)

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label, loss_function=args.loss_function)

logger.debug('Completed  training dataset Preprocess')
AUTOTUNE=1000

#Update status to Training for map function in the preprocess
update_status(True)

def parse_inputs(file_list, file_labels):
    path_ds = tf.data.Dataset.from_tensor_slices(file_list)
    image_ds = path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(file_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # Perhaps the shuffle is the culprit. Changing the size of buffer changes where the scores start to deviate
    ds = image_label_ds.shuffle(buffer_size=100).repeat()
    return ds

train_ds = parse_inputs(train_data.files, train_data.labels)
train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
training_steps = int(train_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Training dataset')


if args.image_dir_validation:
    # Get Validation data
    # Update status to Testing for map function in the preprocess
    update_status(False)
    validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label,
                                 loss_function=args.loss_function)
    logger.debug('Completed test dataset Preprocess')
    validation_ds = parse_inputs(validation_data.files, validation_data.labels)
    validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    logger.debug('Completed Validation dataset')
else:
    validation_ds = None
    validation_steps = None

# Build output name
fname = [args.prefix, args.model_name, args.optimizer, args.lr, args.loss_function, args.train_num_layers]
fname = '_'.join([str(x) for x in fname])
out_dir = os.path.join(args.log_dir, fname)

#Make output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

checkpoint_path = os.path.join(out_dir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

reg_drop_out_per=None
if args.reg_drop_out_per is not None:
    reg_drop_out_per=float(args.reg_drop_out_per)

num_layers=None
if args.train_num_layers is not None:
    num_layers=int(args.train_num_layers)

###############################################################################
# Build the model
###############################################################################
GPU = True
if GPU is True:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        logger.debug('Mirror initialized')
        m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes, num_layers=num_layers, reg_drop_out_per=reg_drop_out_per)
        logger.debug('Model constructed')
        model = m.compile_model(args.optimizer, args.lr, args.loss_function)
        logger.debug('Model compiled')
        model.summary()

        model.save_weights(checkpoint_path.format(epoch=0))
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if not latest:
            model.save_weights(checkpoint_path.format(epoch=0))
            latest = tf.train.latest_checkpoint(checkpoint_dir)

        ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
        logger.debug('Loading initialized model')
        model.load_weights(latest)
        logger.debug('Loading weights from ' + latest)

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
                  shuffle=False,initial_epoch=ini_epoch)
        model.save(os.path.join(out_dir, 'my_model.h5'))
else:
    logger.debug('Not using mirror')
    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=train_data.classes,
                 num_layers=num_layers, reg_drop_out_per=reg_drop_out_per)
    logger.debug('Model constructed')
    model = m.compile_model(args.optimizer, args.lr, args.loss_function)
    logger.debug('Model compiled')
    model.summary()

    model.save_weights(checkpoint_path.format(epoch=0))
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    if not latest:
        model.save_weights(checkpoint_path.format(epoch=0))
        latest = tf.train.latest_checkpoint(checkpoint_dir)

    ini_epoch = int(re.findall(r'\b\d+\b', os.path.basename(latest))[0])
    logger.debug('Loading initialized model')
    model.load_weights(latest)
    logger.debug('Loading weights from ' + latest)

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
