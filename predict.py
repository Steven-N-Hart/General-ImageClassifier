from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import tensorflow as tf
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example
import matplotlib.pylab as plt
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import PIL.Image as Image
import glob

#filepath="/projects/shart/digital_pathology/results/tcga_pten_General-ImageClassifier/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5"
filepath='/projects/shart/digital_pathology/results/General-ImageClassifier/tcga_pten_General-ImageClassifier/train/ResNet50_SGD_0.001-BinaryCrossentropy/my_model.h5'
#input_dir='/projects/shart/digital_pathology/data/TCGA/General-ImageClassifier-Level3_selected_PTEN/train'
input_dir='/projects/shart/digital_pathology/data/TCGA/General-ImageClassifier-Level3_selected_PTEN/val'
files=glob.glob(input_dir+'/*/*png')
new_model = models.load_model(filepath)
IMAGE_SHAPE = (256, 256)
#new_model.summary()
for file in files:
	#print(file)
	#sys.exit(0)
	file_out = Image.open(file)
	file_out = file_out.resize(IMAGE_SHAPE)
	file_out=np.asarray(file_out)
	file_out = np.reshape(file_out,(1,256,256,3))
	result = np.asarray(new_model.predict(file_out))
	#print(result)	
	print(os.path.basename(file)+' '+str((result[0][0]))+' '+str((result[0][1])))
	#sys.exit(0)
