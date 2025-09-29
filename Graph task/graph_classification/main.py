import torch
from model import GNNClassifier
from data_loader import load_data
from train_eval import train, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_types = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
poolings = ['mean', 'max', 'min']

dataset_name = 'PROTEINS' #TUDataset, ZINC

dataset, train_loader, test_loader, task_type = load_data(name=dataset_name)
input_dim = dataset.num_features
output_dim = 1 if task_type == 'regression' else dataset.num_classes

for model_type in model_types:
    for pooling in poolings:
        print(f"\n>> {dataset_name}: {model_type} + {pooling} pooling")
        model = GNNClassifier(model_type, pooling, input_dim, output_dim, task_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        criterion = (
            torch.nn.L1Loss() if task_type == 'regression' else torch.nn.CrossEntropyLoss()
        )

        for epoch in range(1, 51):
            loss = train(model, train_loader, optimizer, criterion, device)
        metric = test(model, test_loader, device)

        if task_type == 'regression':
            print(f"Test MAE: {metric:.4f}")
        else:
            print(f"Test Accuracy: {metric:.4f}")
