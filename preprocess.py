from random import shuffle, choice
import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class Preprocess:

    def __init__(self, directory_path, loss_function=None):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        logger.debug('Initializing Preprocess')
        self.directory_path = directory_path
        self.loss_function = loss_function
        self.classes = self.__get_classes()
        self.files, self.labels, self.label_dict, self.min_images = self.__get_lists()

    def __check_min_image(self, prev, new):
        logger.debug('Counting the number of images')
        if prev is None or prev > new:
            return new
        else:
            return prev

    def __get_classes(self):
        classes = os.listdir(self.directory_path)
        return classes.__len__()

    def __get_lists(self):
        logging.debug('Getting initial list of images and labels')

        files = []
        labels = []
        label_dict = dict()
        label_number = 0
        min_images = None

        classes = os.listdir(self.directory_path)

        for x in classes:
            class_files = os.listdir(os.path.join(self.directory_path, x))
            class_files = [os.path.join(self.directory_path, x, j) for j in class_files]
            class_labels = [label_number for x in range(class_files.__len__())]
            min_images = self.__check_min_image(min_images, class_labels.__len__())
            label_dict[x] = label_number
            label_number += 1
            files.extend(class_files)
            labels.extend(class_labels)

        labels = tf.dtypes.cast(labels, tf.uint8)
        # I noticed that if your loss function expects loss, it has to be one hot, otherwise, it expects an int
        if not self.loss_function.startswith('Sparse'):
            labels = tf.one_hot(labels, classes.__len__())

        return files, labels, label_dict, min_images



def format_example(image_name=None, img_size=256, train=True):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :param train: whether this is for training or not

    :return: image
    """
    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.2)

    image = tf.reshape(image, (img_size, img_size, 3))

    return image
