import argparse
import numpy as np
import torch
import torchvision
from torchvision import models
import random
from collections import Counter
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import wandb
import os
import opacus
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from auditing_utils import find_O1_pred, generate_auditing_data, find_O1_pred_v2, insert_canaries
from utils import torch_to_np, np_to_torch, save_plot, save_data_instance, Net, save_model, CNNSmall, CanariesDataset

parser = argparse.ArgumentParser(description='MNISTSim')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--N', default=50, type=int)
parser.add_argument('--data_gen_noise', default=0.3, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--eval_every', default=1, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--audit', action='store_true')
parser.add_argument('--audit_proportion', default=0.1, type=float)
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--data_path', default='/home/qiaoyuet/project/data', type=str)
parser.add_argument('--save_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim', type=str)
parser.add_argument('--l2_reg', default=0.0, type=float)
parser.add_argument('--target_epsilon', default=-1., type=float)
parser.add_argument('--delta', default=1e-5, type=float)
parser.add_argument('--dp_C', default=1., type=float)
parser.add_argument('--dp_noise', default=-1., type=float)
parser.add_argument('--non_priv', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def eval_model(net, loader, audit=False):
    net.eval()
    if audit:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # get per-sample loss for audit
        all_losses = []
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        all_losses = 0
    total, correct = 0, 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        t_loss = criterion(outputs, labels)
        t_loss = torch_to_np(t_loss)
        if audit:
            all_losses.extend(t_loss)
        else:
            all_losses += float(t_loss)
            all_losses /= len(loader.dataset)
    acc = correct / total
    return acc, all_losses


def train(train_loader, test_loader, mem_loader, non_mem_loader):
    net = CNNSmall().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_loader, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_loader, audit=True)

    step_counter = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.train()
        for i, data in tqdm(enumerate(train_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step_counter += 1

            if step_counter % args.eval_every == 0:
                # train_stats
                train_acc, _ = eval_model(net, train_loader, audit=False)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'train_loss': float(torch_to_np(loss)), 'train_acc': float(train_acc)
                }
                if not args.debug:
                    wandb.log(train_metric)

                # test_stats
                test_acc, t_loss = eval_model(net, test_loader, audit=False)
                test_metric = {
                    'test_acc': float(test_acc),
                    'test_loss': float(np.mean(np.array(t_loss)))
                }
                if not args.debug:
                    wandb.log(test_metric)

                # audit_stats
                if args.audit:
                    _, cur_mem_losses = eval_model(net, mem_loader, audit=True)
                    _, cur_non_mem_losses = eval_model(net, non_mem_loader, audit=True)
                    mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                    non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                    audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                    if not args.debug:
                        wandb.log(audit_metrics)


def train_priv(train_loader, test_loader, mem_loader, non_mem_loader):
    net = CNNSmall().to(device)
    net = ModuleValidator.fix(net)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

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
        _, init_mem_losses = eval_model(net, mem_loader, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_loader, audit=True)

    step_counter = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.train()
        for i, data in tqdm(enumerate(train_loader)):
            net.train()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step_counter += 1

            if step_counter % args.eval_every == 0:
                # train_stats
                epsilon = privacy_engine.get_epsilon(args.delta)
                train_acc, _ = eval_model(net, train_loader, audit=False)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'train_loss': float(torch_to_np(loss)), 'train_acc': float(train_acc),
                    'dp_eps': epsilon
                }
                if not args.debug:
                    wandb.log(train_metric)

                # test_stats
                test_acc, t_loss = eval_model(net, test_loader, audit=False)
                test_metric = {
                    'test_acc': float(test_acc),
                    'test_loss': float(np.mean(np.array(t_loss)))
                }
                if not args.debug:
                    wandb.log(test_metric)

                # audit_stats
                if args.audit:
                    _, cur_mem_losses = eval_model(net, mem_loader, audit=True)
                    _, cur_non_mem_losses = eval_model(net, non_mem_loader, audit=True)
                    mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                    non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                    audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                    if not args.debug:
                        wandb.log(audit_metrics)


def main():
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)

    # load mnist data
    train_data = torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    test_data = torchvision.datasets.MNIST(args.data_path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

    # optional: subset mnist data
    targets = train_data.targets
    target_indices = np.arange(len(targets))
    train_1_idx, train_2_idx = train_test_split(target_indices, train_size=0.1, stratify=targets)
    train_data_sub = Subset(train_data, train_1_idx)
    train_data_sub.targets = train_data.targets[train_1_idx]
    train_data_sub.data = train_data.data[train_1_idx]

    train_loader = torch.utils.data.DataLoader(train_data_sub, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    if args.audit:
        # create canaries
        targets = train_data_sub.targets
        target_indices = np.arange(len(targets))
        train_idx, canary_idx = train_test_split(target_indices, train_size=(1-args.audit_proportion), stratify=targets)
        canary_sub = Subset(train_data_sub, canary_idx)
        orig_targets = train_data_sub.targets[canary_idx]
        idx = torch.randperm(orig_targets.nelement())
        new_targets = orig_targets.view(-1)[idx].view(orig_targets.size())
        canary_sub.targets = new_targets
        canary_sub.data = train_data_sub.data[canary_idx]
        new_train_sub = Subset(train_data_sub, train_idx)
        mem_data, non_mem_data = torch.utils.data.random_split(
            canary_sub, [0.5, 0.5],
            generator=torch.Generator().manual_seed(1024))
        new_train_data = torch.utils.data.ConcatDataset([new_train_sub, mem_data])
        train_loader = torch.utils.data.DataLoader(new_train_data, batch_size=args.batch_size, shuffle=True)
        mem_loader = torch.utils.data.DataLoader(mem_data, batch_size=args.batch_size, shuffle=True)
        non_mem_loader = torch.utils.data.DataLoader(non_mem_data, batch_size=args.batch_size, shuffle=True)
        if not args.debug:
            wandb.log({'num_mem': len(mem_data), 'num_non_mem': len(non_mem_data)})
    else:
        mem_loader, non_mem_loader = None, None

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.non_priv:
        train(train_loader, test_loader, mem_loader, non_mem_loader)
    else:
        train_priv(train_loader, test_loader, mem_loader, non_mem_loader)


if __name__ == '__main__':
    main()
