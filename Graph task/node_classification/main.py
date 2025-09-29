import argparse
import time
import torch
from datasets.load_data import load_dataset
from models.gcn import GCN
from models.gat import GAT
from models.sage import GraphSAGE
from models.gin import GIN
from train import train_full, train_mini_batch, evaluate
from torch_geometric.loader import NeighborLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def get_model(name, in_dim, hidden, out_dim):
    name = name.lower()
    if name == 'gcn':
        return GCN(in_dim, hidden, out_dim)
    elif name == 'gat':
        return GAT(in_dim, hidden, out_dim)
    elif name in ('sage', 'graphsage'):
        return GraphSAGE(in_dim, hidden, out_dim)
    elif name == 'gin':
        return GIN(in_dim, hidden, out_dim)
    else:
        raise ValueError(f'Unknown model: {name}')

@torch.no_grad()
def visualize(model, data, device, use_pred=True, method='tsne'):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    logits = out
    labels = logits.argmax(dim=1) if use_pred else data.y
    labels = labels.cpu().numpy()

    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)

    embeddings = reducer.fit_transform(logits.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=10)
    plt.title('Node Classification Visualization')
    plt.colorbar(scatter)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cora', help='Dataset: Cora, Citeseer, Flickr')
    parser.add_argument('--model', default='gcn', help='Model: gcn, gat, sage, gin')
    parser.add_argument('--batch', action='store_true', help='Use NeighborLoader mini-batch')
    parser.add_argument('--visualize', action='store_true', help='Visualize classification result')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    model = get_model(args.model, dataset.num_features, 64, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if args.batch:
        train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=1024, input_nodes=data.train_mask)
        t0 = time.time()
        for epoch in range(1, 101):
            loss = train_mini_batch(model, train_loader, optimizer, device)
        runtime = time.time() - t0
    else:
        t0 = time.time()
        for epoch in range(1, 201):
            loss = train_full(model, data, optimizer, device)
        runtime = time.time() - t0

    accs = evaluate(model, data, device)
    print(f"Train / Val / Test Accuracy: {accs}")
    print(f"Total Training Time: {runtime:.2f} seconds")

    if args.visualize and not args.batch:
        print("Visualizing classification result...")
        visualize(model, data, device, use_pred=True, method='tsne')
    elif args.visualize and args.batch:
        print("Visualization is only supported in full-graph mode (not mini-batch).")

if __name__ == '__main__':
    main()
