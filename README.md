# CSGCL: Community Strength Enhanced Graph Contrastive Learning
PyTorch implementation for IJCAI 2023 Under Review Paper CSGCL:Community Strength Enhanced Graph Contrastive Learning,The implementation is based on WWW 2021 Paper Graph Contrastive Learning with Adaptive Augmentation implementation(https://github.com/CRIPAC-DIG/GCA), much apperciate to them!
# Requirements
* Python 3.8.8
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.22.4





# Overview of the proposed CSGCL framework
 ![Image text](https://github.com/HanChenHUSTAIA/CSGCL/blob/main/CSGCL%20Model/CSGCL.jpg)




# Datasets



<div align="center">
<table border="1" cellspacing="0">
 
| Dataset | Type | Nodes |Edges| Attributes| Classes |
|  :-:   | :-:   | :-:   | :-:   |  :-:   | :-:  |
| WikiCS| reference |11,701 |216,123 |300 |10 |
|Amazon-Photo |co-purchase | 7,487 | 119,043 | 745|  8
| Amazon-Computers | co-purchase | 13,381 | 245,778|  767 | 10
| Coauthor-CS | co-author | 18,333|  81,894|  6,805 | 15

</table>
</div>

Run the code in Examples,the datasets our paper used will auto download to new folder "datasets",for example "./datasets/WikiCS".




# Examples
The four datasets' best hyperparameter we getted from the parameter sensitively experiment has been written into json format file respectively,for example,if you want to
train and evalute WikiCS datasets by our best hyperparater,you can run the code like below.
<pre class="md-fences md-end-block md-fences-with-lineno ty-contain-cm modeLoaded" spellcheck="false" lang="java" cid="n55" mdtype="fences" style="box-sizing: border-box; overflow: visible; font-family: var(--monospace); font-size: 0.9em; display: block; break-inside: avoid; text-align: left; white-space: normal; background-image: inherit; background-position: inherit; background-size: inherit; background-repeat: inherit; background-attachment: inherit; background-origin: inherit; background-clip: inherit; background-color: rgb(248, 248, 248); position: relative !important; border: 1px solid rgb(231, 234, 237); border-radius: 3px; padding: 8px 4px 6px 0px; margin-bottom: 0px; margin-top: 15px; width: inherit;"> python train.py --dataset WikiCS --param local:wikics.json </pre> 

 
if you want to change the hyperparameter,for example,if you want to change the first view's edge drop rate to 0.4,you can run the code like below.
 <pre class="md-fences md-end-block md-fences-with-lineno ty-contain-cm modeLoaded" spellcheck="false" lang="java" cid="n55" mdtype="fences" style="box-sizing: border-box; overflow: visible; font-family: var(--monospace); font-size: 0.9em; display: block; break-inside: avoid; text-align: left; white-space: normal; background-image: inherit; background-position: inherit; background-size: inherit; background-repeat: inherit; background-attachment: inherit; background-origin: inherit; background-clip: inherit; background-color: rgb(248, 248, 248); position: relative !important; border: 1px solid rgb(231, 234, 237); border-radius: 3px; padding: 8px 4px 6px 0px; margin-bottom: 0px; margin-top: 15px; width: inherit;">  python train.py --dataset WikiCS --param local:wikics.json --drop_edge_rate_1 0.4</pre> 

# Results



<div align="center">
<table border="1" cellspacing="0">
 
| Dataset |WikiCS| Amazon-Computers |Amazon-Photo|Coauthor-CS|
|  :-:   | :-:   | :-:   | :-:   |  :-:   | :-:  |
| Raw (LogReg)| 71.85±0.00 |73.25±0.00 |79.02±0.00 |89.64±0.00
|DeepWalk (w/o X) |73.84±0.16 | 85.77±0.58 | 89.06±0.43 | 84.71±0.35 
| DeepWalk (w/ X)  |  77.21±0.03 |86.28±0.07 | 90.05±0.08 | 87.70±0.04
| Node2Vec |75.52±0.17 |86.19±0.26 | 88.86±0.43 | 86.27±0.22 
| GCN | 78.02±0.51 | 87.79±0.36 | 91.82±0.01 | 93.06±0.00
| GAT | 77.62±0.69 | 88.64±0.63 | 92.16±0.47 | 91.49±0.30
| GRACE | co-author | 18,333|  81,894|  6,805 
| GAE  | co-author | 18,333|  81,894|  6,805 
|VGAE | co-author | 18,333|  81,894|  6,805 
| DGI| co-author | 18,333|  81,894|  6,805 
| MVGRL | co-author | 18,333|  81,894|  6,805 
| GRACE | co-author | 18,333|  81,894|  6,805 
| AFGRL | co-author | 18,333|  81,894|  6,805 
| gCooL-best  | co-author | 18,333|  81,894|  6,805 
| CSGCL | co-author | 18,333|  81,894|  6,805 

Raw (LogReg) & $\mathbf{X}$ & - & 71.85±0.00 & 73.25±0.00 & 79.02±0.00 & 89.64±0.00\\ 
DeepWalk (w/o $\mathbf{X}$) & $\mathbf{A}$ & node & 73.84±0.16 & 85.77±0.58 & 89.06±0.43 & 84.71±0.35 \\ 
Node2Vec & $\mathbf{A}$ & node & 75.52±0.17 & 86.19±0.26 & 88.86±0.43 & 86.27±0.22\\ 
DeepWalk (w/ $\mathbf{X}$) & $\mathbf{X},\mathbf{A}$ & node & 77.21±0.03 & 86.28±0.07 & 90.05±0.08 & 87.70±0.04 \\ 
CPNE & $\mathbf{A}$ & community & 65.53±0.58 & 73.66±0.83 & 82.39±1.23 & OOM \\
\hline\hline
%MVGRL
GAE & $\mathbf{X},\mathbf{A}$ & node & 70.15±0.01 & 85.27±0.19 & 91.62±0.13 & 90.01±0.71\\
VGAE & $\mathbf{X},\mathbf{A}$ & node & 75.63±0.19 & 86.37±0.21 & 92.20±0.11 & 92.11±0.09\\
DGI & $\mathbf{X},\mathbf{A}$ & node & 75.35±0.14 & 83.95±0.47 & 91.61±0.22 & 92.15±0.63\\
MVGRL & $\mathbf{X},\mathbf{A}$ & node & 77.52±0.08 & 87.52±0.11 & 91.74±0.07 & 92.11±0.12\\
GRACE & $\mathbf{X},\mathbf{A}$ & node & 77.68±0.34 & 88.29±0.11 & 92.52±0.34 & 92.50±0.08\\
GCA-best & $\mathbf{X},\mathbf{A}$ & node & 78.20±0.04 & 87.99±0.13 & 92.06±0.27 & 92.81±0.19\\
% GCA-best* & $\mathbf{X},\mathbf{A}$ & node & 78.35±0.05 & 87.85±0.31 & 92.53±0.16 & 93.10±0.01\\
AFGRL & $\mathbf{X},\mathbf{A}$ & node & 77.62±0.49 & 89.88±0.33 & 93.22±0.28 & 93.27±0.17\\
gCooL-best & $\mathbf{X},\mathbf{A}$ & community & 78.20±0.09 & 88.67±0.10 & 92.84±0.20 & 92.75±0.01\\
%COSTA & $\mathbf{X},\mathbf{A}$ & 79.12±0.02 & 88.32±0.03 & 92.56±0.45 & 92.94±0.10\\
%ProGCL & $\mathbf{X},\mathbf{A}$ & 78.45±0.04 & 89.55±0.16 & 93.64±0.13 & 93.67±0.12\\
CSGCL (ours) & $\mathbf{X},\mathbf{A}$ & community & \textbf{78.60±0.13} & \textbf{90.17±0.17} & \textbf{93.32±0.21} & \textbf{93.55±0.12}\\
\hline\hline
GCN & $\mathbf{X},\mathbf{A},\mathbf{Y}$ & - & 78.02±0.51 & 87.79±0.36 & 91.82±0.01 & 93.06±0.00\\
GAT & $\mathbf{X},\mathbf{A},\mathbf{Y}$ & - & 77.62±0.69 & 88.64±0.63 & 92.16±0.47 & 91.49±0.30\\\hline

</table>
</div>
