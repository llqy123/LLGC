import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from utils import load_airport_data, fully_gen_rand_split
from model import LLGC, PageRankAgg


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="airport",
                    help='Dataset to use.')
parser.add_argument('--lr', type=float, default=1.0,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1.1e-10,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj, NormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')                    
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--trials', type=int, default=10, help='Run multiple trails for fair evaluation')

parser.add_argument('--K', type=int, default=9, help='number of propagation steps.')
parser.add_argument('--drop_out', type=float, default=0.0, help='Hyperbolic linear drop_out')
parser.add_argument('--use_bias', type=int, default=0, help='Hyperbolic linear bias(1 or 0)')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def main(args):
    # load data
    data_path = r'data/airport'
    adj, features, labels = load_airport_data(args.dataset, data_path, args.normalization, args.cuda)
    split_path = f'data/data_splits/{args.dataset}_idx.pt'
    try:
        idx_train, idx_val, idx_test = torch.load(split_path)
    except:
        idx_train, idx_val, idx_test = fully_gen_rand_split(len(labels), val_ratio=0.15, test_ratio=0.15)
        torch.save([idx_train, idx_val, idx_test], split_path)

    sgconv = PageRankAgg(K=args.K, alpha=args.alpha).to("cuda")
    x_gconv, precompute_time = sgconv(features, adj._indices(), None)
    model = LLGC(x_gconv.size(1), labels.max().item() + 1, args.drop_out, args.use_bias)
    model = model.cuda() if args.cuda else model

    # train logistic regression and collect test accuracy
    model, train_time = train(model, x_gconv[idx_train], labels[idx_train], idx_train, args.epochs, args.weight_decay,
                              args.lr)
    acc_test = test(model, x_gconv[idx_test], labels[idx_test])

    print("Test accuracy: {:.4f},  pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(acc_test, precompute_time, train_time, precompute_time+train_time))

    return acc_test

def test(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

def accuracy(output, labels):
    preds = output.argmax(dim=1).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)

def train(model, train_features, train_labels, idx_train, epochs=100, weight_decay=5e-6, lr=0.0002):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t
    return model, train_time

accu_acc = []
for _ in range(args.trials):
    acc_test = main(args)
    accu_acc.append(acc_test)

accu_acc = np.array(accu_acc)
acc_mean, acc_std = accu_acc.mean(), accu_acc.std()

print('='*20)
print(f'Dataset: {args.dataset} Test accuracy of {args.trials} runs: mean {acc_mean:.5f}, std {acc_std:.5f}')