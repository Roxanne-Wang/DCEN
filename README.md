# DCEN

This is the code of "Task-Independent Knowledge Makes for Transferable Representations for Generalized Zero-Shot Learning" in AAAI2021. In this paper, we propose a novel Dual-Contrastive Embedding Network (DCEN) that simultaneously learns taskspecific and task-independent knowledge via semantic alignment and instance discrimination.

![image](https://user-images.githubusercontent.com/58110770/110479539-7b26a980-8120-11eb-913d-a435c707be16.png)


# Quick Start

This code is for awa2 dataset.

- Install PyTorch=0.4.1.
- Install dependencies: pip install -r requirements.txt

## Data Preparation
You need to download images of animals with attributes 2 (awa2) dataset.

## Train

We adopt a two-step training strategy.
Specify the image data path, save path in scripts/1.sh. Then train DCEN on one Titan XP GPU:

``` 
cd scripts
bash 1.sh

``` 

## Test


Evaluating our model (https://drive.google.com/file/d/1Mjw9kSvBpF7wcFkChMaG6uFZ9XY-7y2c/view?usp=sharing) or your own model.
Specify the image data path, model path in scripts/2.sh.

``` 
cd scripts
bash 2.sh

```
