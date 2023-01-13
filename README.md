# CSGCL
PyTorch implementation for IJCAI 2023 Under Review Paper Community Strength Enhanced Graph Contrastive Learning,The implementation is based on WWW 2021 Paper Graph Contrastive Learning with Adaptive Augmentation implementation(https://github.com/CRIPAC-DIG/GCA), much apperciate to them!
# Requirements
* Python 3.8.8
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.22.4
# Datasets
* WikiCS
* Amazon-Computers
* Amazon-Photo
* Coauthor-CS

Run the turtorial code below,the datasets our paper used will auto download to new folder "datasets",for example "./datasets/WikiCS".

# Examples
The best hyperparameter we used has been into json format to each dataset respectively.
```python train.py --dataset WikiCS --param local:wikics.json --device cuda:0 --num_epochs 5000 --loss_scheme mod --beta 10 --start_ep 1000```

if you want to change the hyperparameter,for example,you want to change the first view's edge drop rate to 0.4,you can
```python train.py --dataset WikiCS --param local:wikics.json --device cuda:0 --num_epochs 5000 --drop_feature_thresh 0.7  --drop_edge_rate_1 0.4 --loss_scheme mod --beta 10 --start_ep 1000```
