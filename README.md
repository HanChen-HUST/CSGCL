# CSGCL: Community-Strength-Enhanced Graph Contrastive Learning
PyTorch implementation for IJCAI 2023 Main Track Paper "CSGCL: Community-Strength-Enhanced Graph Contrastive Learning" ([arXiv:2305.04658](https://arxiv.org/abs/2305.04658)).

The code is based on the implementation of [GCA](https://github.com/CRIPAC-DIG/GCA).

We also have a Chinese introduction blog on [Zhihu](https://zhuanlan.zhihu.com/p/628116694).


# Dependencies
* Python 3.8.8
* PyTorch 1.8.1
* torch_geometric 2.0.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.22.4

# Quick Start
The best hyperparameters for node classification (as reported in Appendix C.2 of the paper) can be found in `./param`, which will be directly loaded by `--param`:

~~~
python train.py --dataset WikiCS --param local:wikics.json
~~~

You can change the parameter by either .json files (NOT RECOMMENDED) or simply add it to the command, for example:

```shell
python train.py --dataset WikiCS --param local:wikics.json --num_epochs 5000
```

# Results

All experiments are conducted on an 11GB NVIDIA GeForce GTX 1080 Ti GPU with CUDA . The node classification results are shown below.

|                     | Wiki-CS    | Computers  | Photo      | Coauthor-CS |
| ------------------- | ---------- | ---------- | ---------- | ----------- |
| **GCA** (best conf) | 78.20±0.04 | 87.99±0.13 | 92.06±0.27 | 92.81±0.19  |
| **CSGCL**           | 78.60±0.13 | 90.17±0.17 | 93.32±0.21 | 93.55±0.12  |

# Citing

Please cite our paper for your research if it helps:

~~~latex
@article{csgcl,
  title={CSGCL: Community-Strength-Enhanced Graph Contrastive Learning}, 
  author={Han, Chen and Ziwen, Zhao and Yuhua, Li and Yixiong, Zou and Ruixuan, Li and Rui, Zhang},
  journal={CoRR},
  volume={abs/2305.04658},
  year={2023}
}
~~~

