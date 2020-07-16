#!/usr/bin/env bash
#$ -N GIC
#$ -q gpu
#$ -l gpu=1
#$ -o outlog_log
#$ -e errlog_log
#$ -M hart.steven@mayo.edu
#$ -m ae
#$ -notify
#$ -V
#$ -cwd
#$ -l h_vmem=250G

TRAIN_DATA_FOLDER=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/BRAF_NEW/tfrecord_level2_img/train
VAL_DATA_FOLDER=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/BRAF_NEW/tfrecord_level2_img/val
python train.py -t ${TRAIN_DATA_FOLDER} -v ${VAL_DATA_FOLDER} -b 6 -r 0.01 -m DenseNet201 -L CategoricalCrossentropy -P BRAF -N 15 -e 20
