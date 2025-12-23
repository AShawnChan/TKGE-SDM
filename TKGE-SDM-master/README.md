# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

This is the official code release of the following paper: 

Hong Yao, Zhenglong Chen, Diange Zhou and Shengwen Li: Framework for Species Spatiotemporal Evolution Prediction Based on the Fusion of Temporal Knowledge Graph Embedding and Historical Migration Semantics

<img src="https://github.com/AShawnChan/TKGE-SDM/TKGE-SDM.jpg" alt="regcn_architecture" width="700" class="center">

## Quick Start

### Environment variables & dependencies
```
conda create -n TKGE-SDM python=3.7

conda activate TKGE-SDM

pip install -r requirement.txt
```

### Train models
```
cd src
python main.py -d SDTKG --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=5 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 5
```