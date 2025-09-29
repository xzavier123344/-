from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.utils import train_test_split_edges

def load_dataset(name):
    if name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root=f'./data/{name}', name=name)
    elif name == 'Flickr':
        dataset = Flickr(root='./data/Flickr')
    else:
        raise ValueError("Unsupported dataset.")
    data = dataset[0]
    data = train_test_split_edges(data)
    return data