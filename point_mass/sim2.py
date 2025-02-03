import argparse
import numpy as np
import torch
import random
from collections import Counter
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
import math
import wandb
import os
import opacus
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from auditing_utils import find_O1_pred, generate_auditing_data, find_O1_pred_v2, insert_canaries
from utils import torch_to_np, np_to_torch, save_plot, save_data_instance, save_model

parser = argparse.ArgumentParser(description='MemoSim')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--N', default=50, type=int)
parser.add_argument('--n_features', default=10, type=int)
parser.add_argument('--data_gen_noise', default=0.3, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--audit_size', default=0.20, type=float)
parser.add_argument('--eval_every', default=1, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--audit', action='store_true')
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--save_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim', type=str)
parser.add_argument('--l2_reg', default=0.0, type=float)
parser.add_argument('--target_epsilon', default=-1., type=float)
parser.add_argument('--delta', default=1e-5, type=float)
parser.add_argument('--dp_C', default=1., type=float)
parser.add_argument('--dp_noise', default=-1., type=float)
parser.add_argument('--non_priv', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(args.n_features, 64)
        self.hidden2 = torch.nn.Linear(64, 128)
        self.hidden3 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


class NetSmall(torch.nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.hidden1 = torch.nn.Linear(args.n_features, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.output(x)
        return x


def sim_data(args):
    # fix seed in data generation, change seed when doing algorithm runs
    random.seed(42)
    np.random.seed(42)
    # generate data from linear model y = Xw + noise, noise ~ N(0, sigma^2)
    n_samples, n_features = args.N, args.n_features
    sigma = args.data_gen_noise
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(1, n_features)
    y = X @ w_true.T + (np.random.randn(n_samples) * sigma).reshape(-1, 1)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)

    mem_x, mem_y, non_mem_x, non_mem_y = None, None, None, None
    if args.audit:
        train_data = list(zip(train_x, train_y))
        train_data, mem_data, non_mem_data = generate_auditing_data(train_data, audit_size=int(args.N * args.audit_size), seed=42)
        train_x = [i[0] for i in train_data]
        train_y = [i[1] for i in train_data]
        mem_x = np.array([i[0] for i in mem_data])
        mem_y = np.array([i[1] for i in mem_data])
        non_mem_x = np.array([i[0] for i in non_mem_data])
        non_mem_y = np.array([i[1] for i in non_mem_data])
        if not args.debug:
            wandb.log({'num_member': len(mem_data), 'num_non_member': len(non_mem_data)})

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y


@torch.no_grad()
def eval_model(net, test_x, test_y, criterion, audit=False):
    net.eval()
    if audit:
        criterion = torch.nn.MSELoss(reduction='none')  # get per-sample loss for audit
    t_x = np_to_torch(test_x).to(device)
    t_y = np_to_torch(test_y).to(device)
    y_pred = net(t_x)
    t_loss = criterion(y_pred, t_y)
    t_loss = torch_to_np(t_loss)
    y_pred = torch_to_np(y_pred)
    return y_pred, t_loss


def train(train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y):
    # net = Net().to(device)
    net = NetSmall().to(device)
    if not args.debug:
        wandb.watch(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    x = np_to_torch(train_x)
    y = np_to_torch(train_y)

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)
        out_file = open(os.path.join(args.save_path, args.exp_name, 'mia_predictions.txt'), 'w')

    for epoch in range(args.n_epoch):
        net.train()
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0:
            metrics = {}
            # train_stats
            train_metric = {
                'epoch': epoch,
                'train_loss': float(torch_to_np(loss))
            }
            metrics.update(train_metric)

            # test_stats
            y_pred, t_loss = eval_model(net, test_x, test_y, criterion)
            test_metric = {
                'test_loss': float(t_loss)
            }
            metrics.update(test_metric)

            # save_plot
            if not args.debug and not args.no_plot:
                save_plot(train_x, train_y, net, epoch, args.save_path, args.exp_name)
                save_model(net, epoch, args.save_path, args.exp_name)

            # audit_stats
            if args.audit:
                _, cur_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
                _, cur_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)
                mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                metrics.update(audit_metrics)
                out_file.write(" ".join(audit_metrics['mia_predictions']) + "\n")

            if not args.debug:
                wandb.log(metrics)

    if args.audit: out_file.close()


def train_priv(train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y):
    net = Net().to(device)
    net = ModuleValidator.fix(net)
    if not args.debug:
        wandb.watch(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    x = np_to_torch(train_x)
    y = np_to_torch(train_y)
    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

    privacy_engine = PrivacyEngine(accountant='prv')

    assert args.dp_noise > -1 or args.target_epsilon > -1
    if args.dp_noise > -1:
        net, optimizer, train_loader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=args.dp_C,
            noise_multiplier=args.dp_noise
        )
        if not args.debug: wandb.log({'dp_noise_multiplier': args.dp_noise})
    else:
        net, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.n_epoch,
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            max_grad_norm=args.dp_C,
        )
        if not args.debug: wandb.log({'dp_noise_multiplier': optimizer.noise_multiplier})

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)

    for epoch in range(args.n_epoch):
        # print('train')
        net.train()
        optimizer.zero_grad()
        outputs = net(x.unsqueeze(1))
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0:
            metrics = {}
            # print('eval')
            # train_stats
            epsilon = privacy_engine.get_epsilon(args.delta)
            train_metric = {
                'epoch': epoch,
                'train_loss': float(torch_to_np(loss)),
                'dp_eps': epsilon
            }
            metrics.update(train_metric)

            # test_stats
            y_pred, t_loss = eval_model(net, test_x, test_y, criterion)
            test_metric = {
                'test_loss': float(t_loss)
            }
            metrics.update(test_metric)

            # save_plot
            if not args.debug and not args.no_plot:
                save_plot(train_x, train_y, net, epoch, args.save_path, args.exp_name)
                save_model(net, epoch, args.save_path, args.exp_name)

            # audit_stats
            if args.audit:
                _, cur_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
                _, cur_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)
                mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                metrics.update(audit_metrics)

            if not args.debug:
                wandb.log(metrics)


def main():
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)

    train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y = sim_data(args)
    if not args.debug and not args.no_plot:
        data_dict = {
            'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y
        }
        if not os.path.isdir(os.path.join(args.save_path, args.exp_name)):
            os.mkdir(os.path.join(args.save_path, args.exp_name))
        save_path = os.path.join(args.save_path, args.exp_name, 'data_instance.pkl')
        save_data_instance(data_dict, save_path)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.non_priv:
        train(train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y)
    else:
        train_priv(train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y)


if __name__ == '__main__':
    main()
