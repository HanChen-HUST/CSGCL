# CSGCL
PyTorch implementation for IJCAI 2023 Under Review Paper Community Strength Enhanced Graph Contrastive Learning,The implementation is based on WWW 2021 Paper Graph Contrastive Learning with Adaptive Augmentation implementation(https://github.com/CRIPAC-DIG/GCA), much apperciate to them!
# Requirements
* Python 3.7.4
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* sklearn 0.24.1
* networkx 2.5.1
* numpy 1.22.4
# Examples
python train.py --dataset WikiCS --param local:wikics.json --device cuda:0 --num_epochs 5000 --drop_feature_thresh 0.7  --drop_edge_rate_1 0.2 --drop_edge_rate_2 0.7 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.2  --loss_scheme mod --beta 10 --start_ep 1000
