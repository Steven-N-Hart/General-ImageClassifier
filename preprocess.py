import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
global tf_image, tf_label, status


class Preprocess:

    def __init__(self, directory_path, filetype, tfrecord_image, tfrecord_label, loss_function=None):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        global tf_image, tf_label
        tf_image = tfrecord_image
        tf_label = tfrecord_label
        logger.debug('Initializing Preprocess')
        self.directory_path = directory_path
        self.filetype = filetype
        self.loss_function = loss_function
        self.classes = self.__get_classes()
        self.tfrecord_image = tfrecord_image
        self.tfrecord_label = tfrecord_label
        self.files, self.labels, self.label_dict, self.min_images, self.filetype, self.tfrecord_image, \
        self.tfrecord_label = self.__get_lists()

    @staticmethod
    def __check_min_image(prev, new):
        logger.debug('Counting the number of images')
        if prev is None:
            return new
        else:
            return prev + new

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
        filetype = self.filetype
        tfrecord_image = self.tfrecord_image
        tfrecord_label = self.tfrecord_label
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
        return files, labels, label_dict, min_images, filetype, tfrecord_image, tfrecord_label


def update_status(stat):
    global status
    status = stat
    return stat


# processing images
def format_example(image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :return: image
    """
    global status
    train = status
    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality=20, max_jpeg_quality=90)
        image = tf.keras.preprocessing.image.random_shear(image, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        image = tf.keras.preprocessing.image.random_zoom(image, 0.9, row_axis=0, col_axis=1, channel_axis=2)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image


# extracting images and labels from tfrecords
def format_example_tf(tfrecord_proto, img_size=256):
    # Parse the input tf.Example proto using the dictionary above.
    # Create a dictionary describing the features.
    global tf_image, tf_label, status
    train = status
    image_feature_description = {
        tf_image: tf.io.FixedLenFeature((), tf.string, ""),
        tf_label: tf.io.FixedLenFeature((), tf.int64, -1),
    }
    parsed_image_dataset = tf.io.parse_single_example(tfrecord_proto, image_feature_description)

    image = parsed_image_dataset[tf_image]
    label = parsed_image_dataset[tf_label]
    label = tf.dtypes.cast(label, tf.uint8)
    label = tf.one_hot(label, 2)
    image = tf.io.decode_png(image, channels=3)
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
    return image, label
