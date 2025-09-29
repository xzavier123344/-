import torch
from data_utils import load_dataset
from models import gcn, gat, graphsage, gin
from fullgraph_train import train_full
from sampler_train import train_sampler

model_dict = {
    'GCN': gcn.GCN,
    'GAT': gat.GAT,
    'GraphSAGE': graphsage.GraphSAGE,
    'GIN': gin.GIN
}

def main(dataset_name='Cora', model_name='GCN', mode='full'):
    data = load_dataset(dataset_name)
    input_dim = data.num_features
    hidden_dim = 64

    model = model_dict[model_name](input_dim, hidden_dim)
    print(f"Training {model_name} on {dataset_name} using {mode}-graph...")

    if mode == 'full':
        train_full(model, data)
    elif mode == 'sampler':
        train_sampler(model, data)
    else:
        raise ValueError("Invalid training mode")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, choices=['full', 'sampler'], default='full')
    args = parser.parse_args()

    main(args.dataset, args.model, args.mode)

