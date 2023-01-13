# CSGCL: Community Strength Enhanced Graph Contrastive Learning
PyTorch implementation for IJCAI 2023 Under Review Paper CSGCL:Community Strength Enhanced Graph Contrastive Learning,The implementation is based on WWW 2021 Paper Graph Contrastive Learning with Adaptive Augmentation implementation(https://github.com/CRIPAC-DIG/GCA), much apperciate to them!
# Requirements
* Python 3.8.8
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.22.4
# Datasets



<div align="center">
 
| Dataset | Type | Nodes |Edges| Attributes| Classes |
|  ----  | ----  | ----  | ----  |  ----  | ----  |
| WikiCS| reference |11,701 |216,123 |300 |10 |
|Amazon-Photo |co-purchase | 7,487 | 119,043 | 745|  8
| Amazon-Computers | co-purchase | 13,381 | 245,778|  767 | 10
| Coauthor-CS | co-author | 18,333|  81,894|  6,805 | 15

 </div>

Run the code in Examples,the datasets our paper used will auto download to new folder "datasets",for example "./datasets/WikiCS".

# Examples
The four datasets' best hyperparameter we getted from the parameter sensitively experiment has been written into json format file respectively,for example,if you want to
train and evalute WikiCS datasets by our best hyperparater,you can run code like below.
<pre class="md-fences md-end-block md-fences-with-lineno ty-contain-cm modeLoaded" spellcheck="false" lang="java" cid="n55" mdtype="fences" style="box-sizing: border-box; overflow: visible; font-family: var(--monospace); font-size: 0.9em; display: block; break-inside: avoid; text-align: left; white-space: normal; background-image: inherit; background-position: inherit; background-size: inherit; background-repeat: inherit; background-attachment: inherit; background-origin: inherit; background-clip: inherit; background-color: rgb(248, 248, 248); position: relative !important; border: 1px solid rgb(231, 234, 237); border-radius: 3px; padding: 8px 4px 6px 0px; margin-bottom: 0px; margin-top: 15px; width: inherit;"> python train.py --dataset WikiCS --param local:wikics.json </pre> 

 
if you want to change the hyperparameter,for example,if you want to change the first view's edge drop rate to 0.4,you can run code like below.
 <pre class="md-fences md-end-block md-fences-with-lineno ty-contain-cm modeLoaded" spellcheck="false" lang="java" cid="n55" mdtype="fences" style="box-sizing: border-box; overflow: visible; font-family: var(--monospace); font-size: 0.9em; display: block; break-inside: avoid; text-align: left; white-space: normal; background-image: inherit; background-position: inherit; background-size: inherit; background-repeat: inherit; background-attachment: inherit; background-origin: inherit; background-clip: inherit; background-color: rgb(248, 248, 248); position: relative !important; border: 1px solid rgb(231, 234, 237); border-radius: 3px; padding: 8px 4px 6px 0px; margin-bottom: 0px; margin-top: 15px; width: inherit;">  python train.py --dataset WikiCS --param local:wikics.json --drop_edge_rate_1 0.4</pre> 
