# DCEN

This is the code of "Task-Independent Knowledge Makes for Transferable Representations for Generalized Zero-Shot Learning" in AAAI2021. In this paper, we propose a novel Dual-Contrastive Embedding Network (DCEN) that simultaneously learns taskspecific and task-independent knowledge via semantic alignment and instance discrimination.
![image](https://user-images.githubusercontent.com/58110770/110479539-7b26a980-8120-11eb-913d-a435c707be16.png)

# Quick Start
- Install PyTorch=0.4.1.
- Install dependencies: pip install -r requirements.txt

## Data Preparation
You need to download animals with attributes 2 datasets.

## Train and test

change the data path in 1.sh. Then train DCEN on one GPU:

``` 
cd scripts
bash 1.sh

``` 


