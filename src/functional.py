import numpy as np
import networkx as nx
import torch
from typing import Sequence
from cdlib import algorithms
from cdlib.utils import convert_graph_formats


def community_detection(name):
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        'edmot': algorithms.edmot,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        'ego-splitting': algorithms.egonet_splitter,
        'nnsed': algorithms.nnsed,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]


def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]


def cav_dense(feature: torch.Tensor,
              node_cs: np.ndarray,
              p: float,
              max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    w = x.t() @ torch.tensor(node_cs).to(feature.device)
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, torch.ones_like(w) * max_threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def cav(feature: torch.Tensor,
        node_cs: np.ndarray,
        p: float,
        max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    device = feature.device
    w = x.t() @ torch.tensor(node_cs).to(device)
    w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes


def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    com_cs: np.ndarray) -> torch.Tensor:
    edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)


def community_strength(graph: nx.Graph,
                       communities: Sequence[Sequence[int]]) -> (np.ndarray, np.ndarray):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs
