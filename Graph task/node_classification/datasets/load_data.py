from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import NormalizeFeatures

def load_dataset(name):
    name = name.lower()
    if name in ['cora', 'citeseer']:
        return Planetoid(root=f'data/{name}', name=name.capitalize(), transform=NormalizeFeatures())
    elif name == 'flickr':
        return Flickr(root='data/flickr', transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {name}")