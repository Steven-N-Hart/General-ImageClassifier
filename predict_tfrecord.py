from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import io
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
import glob

filepath = sys.argv[1]  # 'path/to/my_model.h5'
input_dir = sys.argv[2]  # 'path/to/tfrecord_dir'
output_file = sys.argv[3]  # 'predict_brca1.txt'

myfile = open(output_file, mode='wt')
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)
train_files = glob.glob(input_dir + '/train/*/*tfrecords')
val_files = glob.glob(input_dir + '/val/*/*tfrecords')
files = train_files + val_files

for i in files:
    cd = os.path.dirname(i)
    mut = os.path.basename(cd)
    cd = os.path.dirname(cd)
    strval = os.path.basename(cd)
    filename = ''
    result = ''
    raw_dataset = tf.data.TFRecordDataset(i)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for k, v in example.features.feature.items():
            if k == 'image/encoded':
                stream = io.BytesIO(v.bytes_list.value[0])
                file_out = Image.open(stream)
                fileout = file_out.resize(IMAGE_SHAPE).convert('RGB')
                file_out = np.asarray(fileout)
                file_out = np.reshape(file_out, (1, 256, 256, 3))
                result = np.asarray(new_model.predict(file_out))
            if k == 'image/name':
                filename = v.bytes_list.value[0]

        finalstr = filename.decode("utf-8") + '.' + mut + '.' + strval + '.png'
        myfile.write(finalstr + ' ' + str((result[0][0])) + ' ' + str((result[0][1])) + '\n')

myfile.close()
