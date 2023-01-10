import argparse
import os
import os.path as osp
import socket
import time
from datetime import datetime
import random
import copy

import cdlib
from matplotlib import cm, use as matuse, pyplot as plt

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import transforms as T
from torch_geometric.utils import dropout_adj, degree, to_undirected, to_networkx
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import nni

from src.functional import (getmod,
                            transition,
                            get_edge_weight,
                            drop_feature_by_modularity,
                            drop_feature_by_modularity_dense,
                            )
from simple_param.sp import SimpleParam
from src.model import Encoder, GRACE
from src.functional import (drop_edge_weighted,
                            degree_drop_weights,
                            drop_edge_by_modularity,
                            evc_drop_weights,
                            pr_drop_weights,
                            feature_drop_weights,
                            drop_feature_weighted_2,
                            feature_drop_weights_dense,
                            )
from src.eval import log_regression, MulticlassEvaluator, evaluate_clu
from src.utils import (get_base_model,
                       get_activation,
                       generate_split,
                       compute_pr,
                       eigenvector_centrality,
                       save_model,
                       load_model,
                       remove_model)
from src.dataset import get_dataset
from augment import get_cd_algorithm

torch.set_printoptions(threshold=3000)


def train(epoch):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx:int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    if args.com_drop_edge:
        edge_index_1 = drop_edge_by_modularity(data.edge_index, edge_weight, p=param['drop_edge_rate_1'],
                                               threshold=args.drop_edge_thresh)
        edge_index_2 = drop_edge_by_modularity(data.edge_index, edge_weight, p=param['drop_edge_rate_2'],
                                               threshold=args.drop_edge_thresh)
    else:
        edge_index_1 = drop_edge(1)  # view 1
        edge_index_2 = drop_edge(2)  # view 2
    if args.com_drop_feature:
        if args.dataset == 'WikiCS':
            x_1 = drop_feature_by_modularity_dense(data.x, n_mod, param["drop_feature_rate_1"],
                                                   args.drop_feature_thresh)
            x_2 = drop_feature_by_modularity_dense(data.x, n_mod, param["drop_feature_rate_2"],
                                                   args.drop_feature_thresh)
        else:
            x_1 = drop_feature_by_modularity(data.x, n_mod, param["drop_feature_rate_1"], args.drop_feature_thresh)
            x_2 = drop_feature_by_modularity(data.x, n_mod, param['drop_feature_rate_2'], args.drop_feature_thresh)

    else:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # if args.loss_scheme == "mod":
    #     loss, node_loss, com_loss = model.modularity_loss(z1, z2,
    #                                                       n_mod=n_mod,
    #                                                       ep=epoch,
    #                                                       start_ep=param['start_ep'],
    #                                                       readout=param['readout'],
    #                                                       alpha=param['alpha'],
    #                                                       beta_max=param['beta'],
    #                                                       batch_size=args.batch_size
    #                                                       if args.dataset in ['Coauthor-CS', 'Coauthor-Phy', 'PubMed']
    #                                                       else None)
    if args.loss_scheme == "mod":
        loss = model.modularity_loss(z1, z2,
                                     n_mod=n_mod,
                                     ep=epoch,
                                     start_ep=param['start_ep'],
                                     beta_max=param['beta'],
                                     batch_size=args.batch_size
                                     if args.dataset in ['Coauthor-CS', 'Coauthor-Phy', 'PubMed']
                                     else None)
    else:
        loss = model.loss(z1, z2, batch_size=args.batch_size
        if args.dataset in ['Coauthor-CS', 'Coauthor-Phy', 'PubMed'] else None)

    loss.backward()
    optimizer.step()

    # if args.loss_scheme in ["mod"]:
    #     return loss.item(), node_loss.item(), com_loss.item()
    # else:
    #     return loss.item()
    return loss.item()

 
a=open("./amazon_photo_deepwalk.txt","a")
def test(epoch, final=False, tsne_dir=None):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        #z=torch.load("/home/dhc/ComGCL_deepwalk/compare_model/deepwalk/newpt/coauthor_cs/coauthor_cs1.pt")
        #z=data.x

    res = {}
    if "cls" in args.tasks:
        # split data
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
                micro_f1s.append(cls_acc['micro_f1'])
                macro_f1s.append(cls_acc['macro_f1'])
            acc = sum(accs) / len(accs)
            micro_f1 = sum(micro_f1s) / len(micro_f1s)
            macro_f1 = sum(macro_f1s) / len(macro_f1s)
        else:
            cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)
            acc = cls_acc['acc']
            micro_f1 = cls_acc['micro_f1']
            macro_f1 = cls_acc['macro_f1']
        res["acc"] = acc
        res["micro_f1"] = micro_f1
        res["macro_f1"] = macro_f1
        a.write(str(res["acc"]))
        if final and use_nni:
            nni.report_final_result(acc)
        elif use_nni:
            nni.report_intermediate_result(acc)

    if "clu" in args.tasks:
        # node clustering on a whole graph
        nmi, ari = evaluate_clu(args.dataset, z, data.y, random_state=args.kmeans_seed)
        res["nmi"] = nmi
        res["ari"] = ari
        a.write(str(res["nmi"]))
        a.write("\n")
    if "link" in args.tasks:
        # split data
        data_z = copy.copy(data)
        data_z.x = z
        data_z = data_z.cpu()
        split = T.RandomLinkSplit(num_val=0., num_test=0., is_undirected=True, add_negative_train_samples=True)
        data_z = split(data_z)[0]
        data_z = data_z.cuda(device)

        src = z[data_z.edge_label_index[0]]
        dst = z[data_z.edge_label_index[1]]
        out = (src * dst).sum(dim=-1).view(-1).sigmoid()
        res["auc"] = roc_auc_score(data_z.edge_label.cpu().numpy(), out.cpu().numpy())
        a.write(str(res["auc"] ))
        a.write("\n")
    if "tsne" in args.tasks and tsne_dir is not None and epoch == args.tsne_ep:
        matuse('agg')
        perplexities = [50, 100]
        for perplexity in perplexities:
            tsne = TSNE(
                n_components=2,
                init="pca",
                random_state=args.seed,
                perplexity=perplexity,
                n_jobs=-1
            )
            # draw scatter plots
            Y = data.y.cpu().numpy().astype(int)
            colors = cm.rainbow(np.linspace(0, 1, max(Y) + 1))
            emb = tsne.fit_transform(z.cpu().numpy())
            fig = plt.figure(figsize=(8, 8))
            axes = fig.add_subplot()
            emb_norm = (emb - emb.min(0)) / (emb.max(0) - emb.min(0))

            for i in range(max(Y) + 1):
                axes.scatter(emb_norm[np.nonzero(Y == i), 0], emb_norm[np.nonzero(Y == i), 1],
                             color=colors[i])
            figformat = 'pdf'
            fig.savefig(osp.join(tsne_dir, f"{args.dataset}_{train_scheme}_e={epoch}_p={perplexity}." + figformat),
                        format=figformat)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-t', '--tasks', type=str, default="cls", nargs='+',
                        choices=["cls", "clu", "link", "tsne"])
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)  # for torch
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--kmeans_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--d_num', type=int, default=1024)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--log_dir', type=str, default=None, metavar='TENSORBOARD LOG PATH')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--validate_interval', type=int, default=100)
    parser.add_argument('--tsne_ep', type=int, default=1000)
    parser.add_argument('--tsne_dir', type=str, default='./tsne', metavar='TSNE PDF SAVING PATH')
    parser.add_argument('--cluster_method', type=str, default='leiden')
    parser.add_argument('-e', '--com_drop_edge', action='store_true')
    parser.add_argument('--drop_edge_thresh', type=float, default=1.)
    parser.add_argument('-f', '--com_drop_feature', action='store_true')
    parser.add_argument('--drop_feature_thresh', type=float, default=1.)
    parser.add_argument('--loss_scheme', type=str, default=None,
                        choices=["mod"])
    parser.add_argument('-s', '--save_model', type=str, default=None,
                        metavar='MODEL PATH', dest='model_path')
    parser.add_argument('-r', '--resume_from_checkpoint', type=str, default=None,
                        metavar='MODEL PTH FILE', dest='cp')
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
        'alpha': 0.5,
        'beta': 1.,
        'sigma': 0.25,
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    # comment = args.dataset + \
    #           (f'_node_{param["drop_feature_rate_1"]}_{param["drop_feature_rate_2"]}'
    #            if args.com_drop_feature else '') + \
    #           (f'_edge_{param["drop_edge_rate_1"]}_{param["drop_edge_rate_2"]}'
    #            if args.com_drop_edge else '') + \
    #           (f'_alpha_{param["alpha"]}_beta_{param["beta"]}'
    #            if args.loss_scheme in ["mod"] else '')
    comment = args.dataset + \
              (f'_node_{param["drop_feature_rate_1"]}_{param["drop_feature_rate_2"]}'
               if args.com_drop_feature else '') + \
              (f'_edge_{param["drop_edge_rate_1"]}_{param["drop_edge_rate_2"]}'
               if args.com_drop_edge else '') + \
              (f'_t0_{param["start_ep"]}_beta_{param["beta"]}'
               if args.loss_scheme in ["mod"] else '')
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    train_scheme = ('ComIso' if args.com_drop_edge else 'DropEdge') + ' & ' + \
                   ('ComKnock' if args.com_drop_feature else 'DropFeature')
    if args.loss_scheme == "mod":
        loss_scheme_str = 'Team-Up InfoNCE'
    else:
        loss_scheme_str = 'InfoNCE'
    print(f"training settings: \n"
          f"{train_scheme}\n"
          f"downstream task: {args.tasks}\n"
          f"data: {args.dataset}\n"
          f"community detection method: {args.cluster_method}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"drop edge rate: {param['drop_edge_rate_1']}/{param['drop_edge_rate_2']}\n"
          f"drop node feature rate: {param['drop_feature_rate_1']}/{param['drop_feature_rate_2']}\n"
          f"readout: {param['readout']}\n"
          f"adaptive weight: {param['drop_scheme']}\n"
          f"loss: {loss_scheme_str}\n"
          # f"alpha: {param['alpha']}\n"
          f"beta: {param['beta']}\n"
          # f"sigma: {param['sigma']}\n"
          f"epochs: {param['num_epochs']}\n"
          )

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    if args.cls_seed is not None:
        # for data splitting of cls test
        np.random.seed(args.cls_seed)

    device = torch.device(args.device)

    path = './datasets'
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)

    if args.com_drop_edge or args.com_drop_feature:

        print('Detecting communities...')
        g = to_networkx(data, to_undirected=True)
        start = time.time()
        dc_res = get_cd_algorithm(args.cluster_method)(g)
        end = time.time()
        communities = dc_res.communities
        com = transition(communities, g.number_of_nodes())

        c_mod, n_mod = getmod(g, communities)
        # c_mod = map_modularity(c_mod, param['sigma'])
        edge_weight = get_edge_weight(data.edge_index, com, c_mod)
        com_size = [len(c) for c in communities]
        print(f'Done! \n{len(com_size)} communities detected. \n'
              f'Time consumed: {end - start}s. \n'
              f'The largest community has {com_size[0]} nodes. \n'
              f'{com_size.count(1)} isolated nodes and {com_size.count(2)} node pairs. \n'
              f'Now start training...\n')

    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )
    last_epoch = 0
    if args.cp is not None:
        model, optimizer, last_epoch = load_model(model, optimizer, args.cp, device)

    if not args.com_drop_edge:
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None
    if not args.com_drop_feature:
        # 设置节点特征丢弃的权重
        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)  # 默认权重均为 1

    log = args.verbose.split(',')

    fres = open(f'res/{comment}.csv', 'a')
    best_res = -np.inf
    best_epoch = 0

    log_dir = args.log_dir if args.log_dir is not None else osp.join(
        'runs', args.dataset, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + comment)
    os.makedirs(log_dir, exist_ok=True)
    with SummaryWriter(log_dir=log_dir, comment=comment) as writer:
        epoch = 0
        res = test(epoch, tsne_dir=args.tsne_dir)
        
        if "acc" in res:
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]}, '
                      f'avg_micro_f1 = {res["micro_f1"]}, avg_macro_f1 = {res["macro_f1"]}')
            writer.add_scalar("acc", res["acc"], epoch)
            writer.add_scalar("micro-f1", res["micro_f1"], epoch)
            writer.add_scalar("macro-f1", res["macro_f1"], epoch)
        
        if "nmi" in res and "ari" in res:
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_nmi = {res["nmi"]}, avg_ari = {res["ari"]}')
            writer.add_scalar("nmi", res["nmi"], epoch)
            writer.add_scalar("ari", res["ari"], epoch)
        
        if "auc" in res:
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_auc = {res["auc"]}')
            writer.add_scalar("auc", res["auc"], epoch)
        for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
            # if args.loss_scheme in ["mod"]:
            #     loss, node_loss, com_loss = train(epoch)
            #     if 'train' in log:
            #         print(f'(T) | Epoch={epoch:03d}, total_loss={loss:.4f}, '
            #               f'node_loss={node_loss:.4f}, community_loss={com_loss:.4f}')
            #     writer.add_scalar("train loss", loss, epoch)
            #     writer.add_scalar("node loss", node_loss, epoch)
            #     writer.add_scalar("community loss", com_loss, epoch)
            # else:
            #     loss = train(epoch)
            #     if 'train' in log:
            #         print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
            #     writer.add_scalar("train loss", loss, epoch)

            loss = train(epoch)
            if 'train' in log:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
            writer.add_scalar("train loss", loss, epoch)

            if epoch % args.validate_interval == 0:
                res = test(epoch, tsne_dir=args.tsne_dir)

                if "acc" in res:
                    if 'eval' in log:
                        print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]}, '
                              f'avg_micro_f1 = {res["micro_f1"]}, avg_macro_f1 = {res["macro_f1"]}')
                        if '' in log and epoch % 500 == 0:  # 每 500 轮写入一次最终结果
                            fres.write(f'{comment},{epoch:04d},acc,{res["acc"]}\n')
                            fres.write(f'{comment},{epoch:04d},micro_f1,{res["micro_f1"]}\n')
                            fres.write(f'{comment},{epoch:04d},macro_f1,{res["macro_f1"]}\n')
                    writer.add_scalar("acc", res["acc"], epoch)
                    writer.add_scalar("micro-f1", res["micro_f1"], epoch)
                    writer.add_scalar("macro-f1", res["macro_f1"], epoch)

                if "nmi" in res and "ari" in res:
                    if 'eval' in log:
                        print(f'(E) | Epoch={epoch:04d}, avg_nmi = {res["nmi"]}, avg_ari = {res["ari"]}')
                        if epoch % 500 == 0:  # 每 500 轮写入一次最终结果
                            fres.write(f'{comment},{epoch:04d},nmi,{res["nmi"]}\n')
                            fres.write(f'{comment},{epoch:04d},ari,{res["ari"]}\n')
                    writer.add_scalar("nmi", res["nmi"], epoch)
                    writer.add_scalar("ari", res["ari"], epoch)

                if "auc" in res:
                    if 'eval' in log:
                        print(f'(E) | Epoch={epoch:04d}, avg_auc = {res["auc"]}')
                        if epoch % 500 == 0:  # 每 500 轮写入一次最终结果
                            fres.write(f'{comment},{epoch:04d},auc,{res["auc"]}\n')
                    writer.add_scalar("auc", res["auc"], epoch)

                # saving the best model
                if args.model_path is not None:
                    if "acc" in res:
                        current_metric = "acc"
                    elif "nmi" in res:
                        current_metric = "nmi"
                    elif "auc" in res:
                        current_metric = "auc"
                    else:
                        raise KeyError('No metrics to compare for best model saving.')
                    current_res = round(res[current_metric], 4)

                    if current_res > best_res:
                        if best_epoch > 0:
                            remove_model(best_epoch, comment, args.model_path)
                        best_epoch = epoch
                        best_res = current_res
                        pth = save_model(model, optimizer, epoch, comment, args.model_path)
                        print(f'(E) | Epoch = {epoch}: newly best model '
                              f'({current_metric}={current_res}) saved in \'{pth}\'. ')

    if use_nni:
        res = test(epoch, final=True)

        if 'final' in log:
            print(f'{res}')

    fres.close()
