import argparse
import os.path as osp
import random
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from src.functional import (getmod,
                            transition,
                            get_edge_weight,
                            drop_feature_by_modularity,
                            drop_feature_by_modularity_dense,
                            )
from src.model import Encoder, GRACE
from src.functional import (drop_edge_weighted,
                            drop_edge_by_modularity,
                            )
from src.eval import log_regression, MulticlassEvaluator
from src.utils import (get_base_model,
                       get_activation,
                       generate_split,
                       )
from src.dataset import get_dataset
from src.augment import get_cd_algorithm
def train(epoch):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = drop_edge_by_modularity(data.edge_index, edge_weight, p=param['drop_edge_rate_1'],
                                               threshold=args.drop_edge_thresh)
    edge_index_2 = drop_edge_by_modularity(data.edge_index, edge_weight, p=param['drop_edge_rate_2'],
                                               threshold=args.drop_edge_thresh)
    if args.dataset == 'WikiCS':
        x_1 = drop_feature_by_modularity_dense(data.x, n_mod, param["drop_feature_rate_1"],
                                               args.drop_feature_thresh)
        x_2 = drop_feature_by_modularity_dense(data.x, n_mod, param["drop_feature_rate_2"],
                                               args.drop_feature_thresh)
    else:
        x_1 = drop_feature_by_modularity(data.x, n_mod, param["drop_feature_rate_1"], args.drop_feature_thresh)
        x_2 = drop_feature_by_modularity(data.x, n_mod, param['drop_feature_rate_2'], args.drop_feature_thresh)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    if args.loss_scheme == "mod":
        loss = model.modularity_loss(z1, z2,
                                     n_mod=n_mod,
                                     ep=epoch,
                                     start_ep=param['start_ep'],
                                     beta_max=param['beta'],
                                     batch_size=args.batch_size
                                     if args.dataset in ['Coauthor-CS']
                                     else None)
    else:
        loss = model.loss(z1, z2, batch_size=args.batch_size
        if args.dataset in ['Coauthor-CS'] else None)
    loss.backward()
    optimizer.step()
    return loss.item()
def test(final=False):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    res = {}
    seed = np.random.randint(0, 32767)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1,
                           generator=torch.Generator().manual_seed(seed))
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        micro_f1s, macro_f1s = [], []
        for i in range(20):
            cls_acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)
            accs.append(cls_acc['acc'])
        acc = sum(accs) / len(accs)
    else:
        cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)
        acc = cls_acc['acc']
    res["acc"] = acc
    return res
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)  # for torch
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--validate_interval', type=int, default=100)
    parser.add_argument('--community_detection_method', type=str, default='leiden')
    parser.add_argument('--drop_edge_thresh', type=float, default=1.)
    parser.add_argument('--drop_feature_thresh', type=float, default=1.)
    parser.add_argument('--loss_scheme', type=str, default=None,
                        choices=["mod"])
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
        'readout': 'mean',
        'start_ep': 500,
        'beta': 1.,
        'sigma': 0.25,
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    param = default_param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    comment = args.dataset + \
              (f'_node_{param["drop_feature_rate_1"]}_{param["drop_feature_rate_2"]}'
               + 
              f'_edge_{param["drop_edge_rate_1"]}_{param["drop_edge_rate_2"]}'
                + 
              f'_t0_{param["start_ep"]}_beta_{param["beta"]}'
               if args.loss_scheme in ["mod"] else '')
    if args.device!= 'cpu':
        args.device = 'cuda'
    train_scheme = ('Communal Attribute Voting' ) + ' & ' + \
                   ('Communal Edge Dropping')
    if args.loss_scheme == "mod":
        loss_scheme_str = 'Team-Up InfoNCE'
    else:
        loss_scheme_str = 'InfoNCE'
    print(f"training settings: \n"
          f"data: {args.dataset}\n"
          f"community detection method: {args.community_detection_method}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"drop edge rate: {param['drop_edge_rate_1']}/{param['drop_edge_rate_2']}\n"
          f"drop node feature rate: {param['drop_feature_rate_1']}/{param['drop_feature_rate_2']}\n"
          f"readout: {param['readout']}\n"
          f"adaptive weight: {param['drop_scheme']}\n"
          f"loss: {loss_scheme_str}\n"
          f"beta: {param['beta']}\n"
          f"epochs: {param['num_epochs']}\n"
          )
    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    if args.cls_seed is not None:
        np.random.seed(args.cls_seed)
    device = torch.device(args.device)
    path = './datasets'
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)
    print('Detecting communities...')
    g = to_networkx(data, to_undirected=True)
    dc_res = get_cd_algorithm(args.community_detection_method)(g)
    communities = dc_res.communities
    com = transition(communities, g.number_of_nodes())
    c_mod, n_mod = getmod(g, communities)
    edge_weight = get_edge_weight(data.edge_index, com, c_mod)
    com_size = [len(c) for c in communities]
    print(f'Done! \n{len(com_size)} communities detected. \n'
          f'The largest community has {com_size[0]} nodes. \n'
         )
    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])
    last_epoch = 0
    log = args.verbose.split(',')
    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % args.validate_interval == 0:
            res = test(epoch)
            if "acc" in res:
                if 'eval' in log:
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]}')
