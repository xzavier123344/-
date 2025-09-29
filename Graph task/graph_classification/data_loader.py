from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader

def load_data(name='MUTAG', batch_size=32):
    if name.upper() == 'ZINC':
        dataset = ZINC(root='data/ZINC')
        task_type = 'regression'
    else:
        dataset = TUDataset(root='data/TUDataset', name=name)
        task_type = 'classification'

    dataset = dataset.shuffle()
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, train_loader, test_loader, task_type
