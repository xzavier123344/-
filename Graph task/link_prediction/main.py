import argparse
from train import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['GCN', 'GAT', 'SAGE', 'GIN'], default='GIN')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'Citeseer', 'Flickr'], default='Cora')
    parser.add_argument('--sampling', action='store_true')
    args = parser.parse_args()

    run_experiment(args.model, args.dataset, args.sampling)
