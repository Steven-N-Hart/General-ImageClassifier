from __future__ import absolute_import, division, print_function, unicode_literals
import os
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob
import sys

filepath = sys.argv[1]  # "/path/to/my_model.h5"
input_dir = sys.argv[2]  # '/path/to/test'
files = glob.glob(input_dir + '/*/*png')
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)

for file in files:
	file_out = Image.open(file)
	file_out = file_out.resize(IMAGE_SHAPE)
	file_out = np.asarray(file_out)
	file_out = np.reshape(file_out, (1, 256, 256, 3))
	result = np.asarray(new_model.predict(file_out))
	print(os.path.basename(file) + ' ' + str((result[0][0])) + ' ' + str((result[0][1])))
