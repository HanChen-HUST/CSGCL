import os.path as osp
from torch_geometric.datasets import WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T

def get_dataset(path, name):
    assert name in [ 'WikiCS', 'Coauthor-CS','Amazon-Computers', 'Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name
    root_path = './datasets'
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)


if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    path = './datasets'
    path = osp.join(path, 'WikiCS')
    dataset = TUDataset(root=path, name='tu', transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to('cuda:2')
    print(data)
