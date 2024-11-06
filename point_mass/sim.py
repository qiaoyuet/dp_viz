import argparse
import numpy as np
import torch
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
import math
import wandb
import os

from utils import np_to_torch, torch_to_np
from auditing_utils import find_O1_pred, generate_auditing_data, find_O1_pred_v2, insert_canaries


parser = argparse.ArgumentParser(description='MemoSim')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--N', default=50, type=int)
parser.add_argument('--data_gen_noise', default=0.3, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--eval_every', default=1, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--audit', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--save_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim', type=str)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64)
        self.hidden2 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


def sim_data(args):
    # true function: sin 3x in the range of 0-2
    f = lambda x: np.sin(3 * x)
    train_x = [np.random.uniform(0, 2) for _ in range(args.N)]
    tmp_y = [f(i).item() for i in train_x]
    # data generating function: true func + noise
    train_y = [np.random.normal(i, 0.1, 1)[0].item() for i in tmp_y]

    test_x = [np.random.uniform(0, 2) for _ in range(args.N)]
    test_y = [f(i).item() for i in train_x]

    mem_x, mem_y, non_mem_x, non_mem_y = None, None, None, None
    if args.audit:
        train_data = list(zip(train_x, train_y))
        train_data, mem_data, non_mem_data = generate_auditing_data(train_data, audit_size=args.N, seed=args.seed)
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


def save_plot(train_x, train_y, net, epoch):
    # plot true function
    f = lambda x: np.sin(3 * x)
    x_plot = np.linspace(0, 2, 100)
    actual_y = [f(p).item() for p in x_plot]
    plt.plot(x_plot, actual_y, 'g', label='Actual Function')

    # plot train data
    plt.scatter(train_x, train_y)

    # plot est function
    x_plot = np.linspace(0, 2, 100)
    predicted_y = net(np_to_torch(x_plot).unsqueeze(1)).squeeze()
    plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='Predicted Function')
    plt.legend()
    # plt.show()
    save_path = os.path.join(args.save_path, 'img')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, '{}-e{}.png'.format(args.exp_name, epoch)))
    plt.close()


def eval_model(net, test_x, test_y, criterion, audit=False):
    net.eval()
    if audit:
        criterion = torch.nn.MSELoss(reduction='none')  # get per-sample loss for audit
    with torch.no_grad():
        t_x = np_to_torch(test_x)
        t_y = np_to_torch(test_y)
        y_pred = net(t_x.unsqueeze(1))
        t_loss = criterion(y_pred.squeeze(), t_y)
    return y_pred, t_loss


def train(args, train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y):
    net = Net()
    if not args.debug:
        wandb.watch(net)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    x = np_to_torch(train_x)
    y = np_to_torch(train_y)

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)

    for epoch in range(args.n_epoch):
        net.train()
        optimizer.zero_grad()
        outputs = net(x.unsqueeze(1))
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0:
            # train_stats
            train_metric = {
                'epoch': epoch,
                'train_loss': loss.detach().numpy()
            }
            if not args.debug:
                wandb.log(train_metric)

            # test_stats
            y_pred, t_loss = eval_model(net, test_x, test_y, criterion)
            test_metric = {
                'test_loss': t_loss.detach().numpy()
            }
            if not args.debug:
                wandb.log(test_metric)

            # save_plot
            if not args.debug:
                save_plot(train_x, train_y, net, epoch)

            # audit_stats
            if args.audit:
                _, cur_mem_losses = eval_model(net, mem_x, mem_y, criterion, audit=True)
                _, cur_non_mem_losses = eval_model(net, non_mem_x, non_mem_y, criterion, audit=True)
                mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                if not args.debug:
                    wandb.log(audit_metrics)


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y = sim_data(args)
    train(args, train_x, train_y, test_x, test_y, mem_x, mem_y, non_mem_x, non_mem_y)


if __name__ == '__main__':
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)
    main(args)
