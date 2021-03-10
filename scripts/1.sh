#!/bin/bash

cd ..

MODEL=dcen
SAVE_PATH=/output/
IMG_PATH=/data/cq14/CUB/Animals_with_Attributes2/JPEGImages/

mkdir -p ${SAVE_PATH}
nvidia-smi

python main.py -a ${MODEL} -im ${IMG_PATH} -s ${SAVE_PATH} --opt adam --lr 0.01 -b 120 --cmcn_K 120 --w_Lid 0 --aug v2 --epochs 20 --w_Lsp 0  --Lsi_base  --is_fix   

python main.py -a ${MODEL} -im ${IMG_PATH} -s ${SAVE_PATH} --opt sgd --lr 0.0001  -b 14 --aug v3  --epochs 40 --sem_aug v2  --w_Lsp 1   --resume ${SAVE_PATH}/fix.model   --load_model_part only_si 
