#!/bin/bash

cd ..

MODEL=dcen
SAVE_PATH=/output/
SAVE_MODEL=/model/cq14/final/70.95.model
IMG_PATH=/data/cq14/CUB/Animals_with_Attributes2/JPEGImages/

mkdir -p ${SAVE_PATH}
nvidia-smi

python eval.py -a ${MODEL} -im ${IMG_PATH} -s ${SAVE_PATH}  -b 14  --resume ${SAVE_MODEL}
