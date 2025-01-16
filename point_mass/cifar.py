import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
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
from opacus.schedulers import ExponentialNoise, LambdaNoise, StepNoise
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from auditing_utils import find_O1_pred, generate_auditing_data, find_O1_pred_v2, insert_canaries
from utils import torch_to_np, np_to_torch, save_plot, save_data_instance, Net, save_model, CNNCifar, \
    CanariesDataset, StudentNet, load_model, load_priv_model, StudentCNN

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
parser.add_argument('--train_proportion', default=0.1, type=float)
parser.add_argument('--audit_proportion', default=0.1, type=float)
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--data_path', default='/home/qiaoyuet/project/data', type=str)
parser.add_argument('--save_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim_cifar', type=str)
parser.add_argument('--l2_reg', default=0.0, type=float)
parser.add_argument('--target_epsilon', default=-1., type=float)
parser.add_argument('--delta', default=1e-5, type=float)
parser.add_argument('--dp_C', default=1., type=float)
parser.add_argument('--dp_noise', default=-1., type=float)
parser.add_argument('--non_priv', action='store_true')
parser.add_argument('--save_mode', action='store_true')
parser.add_argument('--save_at_step', default="10,50,100,150,200,600,1000")
parser.add_argument('--distill', action='store_true')
parser.add_argument('--load_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim_mnist', type=str)
parser.add_argument('--load_exp_name', default='tmp', type=str)
parser.add_argument('--load_step', default=-1, type=int)
parser.add_argument('--alpha', default=0.1, type=float, help='distillation loss strength')
# parser.add_argument('--student_model', choices=['1', '2', '3'])
parser.add_argument('--stu_hidden_size', default=32, type=int)
parser.add_argument('--stu_num_hidden', default=32, type=int)
parser.add_argument('--stu_num_out_channels', default=2, type=int)
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


@torch.no_grad()
def eval_ckpt(net, loader):
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    total, correct = 0, 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += int(predicted == labels)
        t_loss = criterion(outputs, labels)
        t_loss = torch_to_np(t_loss)
        all_losses.extend(t_loss)
    acc = correct / total
    return acc, all_losses


@torch.no_grad()
def eval_stu_model(net, loader):
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    all_losses = 0
    total, correct = 0, 0
    for i, data in enumerate(loader):
        inputs, labels = data
        # inputs = inputs.reshape(-1, 28 * 28).to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        t_loss = criterion(outputs, labels)
        t_loss = torch_to_np(t_loss)
        all_losses += float(t_loss)
        all_losses /= len(loader.dataset)
    acc = correct / total
    return acc, all_losses


def train(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader):
    net = CNNCifar().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_loader, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_loader, audit=True)

    if args.save_mode:
        save_steps = [int(item) for item in args.save_at_step.split(',')]

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
                clean_train_acc, _ = eval_model(net, clean_train_loader, audit=False)
                mem_acc, _ = eval_model(net, mem_loader, audit=False)
                non_mem_acc, _ = eval_model(net, non_mem_loader, audit=False)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'train_loss': float(torch_to_np(loss)), 'train_acc': float(train_acc),
                    'clean_train_acc': float(clean_train_acc), 'mem_acc': float(mem_acc),
                    'non_mem_acc': float(non_mem_acc)
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

            if args.save_mode and int(step_counter) in save_steps:
                save_model(net, step_counter, args.save_path, args.exp_name)


def priv_noise_lambda(step):
    if step < 2000:
        return 100
    else:
        return 50


def train_priv(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader):
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
    scheduler = LambdaNoise(optimizer, noise_lambda=priv_noise_lambda)

    if args.audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_loader, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_loader, audit=True)

    if args.save_mode:
        save_steps = [int(item) for item in args.save_at_step.split(',')]

    step_counter = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.train()
        for i, data in tqdm(enumerate(train_loader)):
            net.train()  # opacus issue: if not added every time it returns activation list empty error
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            net.zero_grad()  # opacus issue: memory leak
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_counter += 1

            if step_counter % args.eval_every == 0:
                # train_stats
                epsilon = privacy_engine.get_epsilon(args.delta)
                train_acc, _ = eval_model(net, train_loader, audit=False)
                clean_train_acc, _ = eval_model(net, clean_train_loader, audit=False)
                mem_acc, _ = eval_model(net, mem_loader, audit=False)
                non_mem_acc, _ = eval_model(net, non_mem_loader, audit=False)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'train_loss': float(torch_to_np(loss)), 'train_acc': float(train_acc),
                    'dp_eps': epsilon, 'clean_train_acc': float(clean_train_acc), 'mem_acc': float(mem_acc),
                    'non_mem_acc': float(non_mem_acc), 'schedule_noise_multiplier': float(optimizer.noise_multiplier)
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

            if args.save_mode and int(step_counter) in save_steps:
                save_model(net, step_counter, args.save_path, args.exp_name)


# Distillation loss function for regression
# def distillation_loss(student_outputs, teacher_outputs, true_labels, alpha):
#     ground_truth_loss = F.cross_entropy(student_outputs, true_labels)
#     distill_loss = F.cross_entropy(student_outputs, teacher_outputs)
#     return alpha * ground_truth_loss + (1 - alpha) * distill_loss


def distillation_loss(student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5):
    soft_loss = torch.nn.KLDivLoss(reduction='mean')(
        torch.log_softmax(student_logits / temperature, dim=1),
        torch.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)

    hard_loss = torch.nn.CrossEntropyLoss()(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# Training loop for student model
def train_student(teacher, train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader):
    # input_size = 784  # 28*28 img size
    # hidden_size = args.stu_hidden_size
    # num_hidden = args.stu_num_hidden
    # num_classes = 10
    # student = StudentNet(input_size, num_hidden, hidden_size, num_classes).to(device)

    num_out_channels = args.stu_num_out_channels
    num_classes = 10
    student = StudentCNN(num_out_channels, num_classes).to(device)

    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr)

    step_counter = 0
    for epoch in tqdm(range(args.n_epoch)):
        teacher.eval()
        student.train()
        for i, data in tqdm(enumerate(train_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            # inputs_stu = inputs.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # outputs = student(inputs_stu)
            outputs = student(inputs)
            with torch.no_grad():
                soft_outputs = teacher(inputs)
            loss = distillation_loss(outputs, soft_outputs, labels, temperature=1, alpha=args.alpha)
            loss.backward()
            optimizer.step()
            step_counter += 1

            if step_counter % args.eval_every == 0:
                # train_stats
                train_acc, _ = eval_stu_model(student, train_loader)
                clean_train_acc, _ = eval_stu_model(student, clean_train_loader)
                mem_acc, _ = eval_stu_model(student, mem_loader)
                non_mem_acc, _ = eval_stu_model(student, non_mem_loader)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'stu_train_loss': float(torch_to_np(loss)), 'stu_train_acc': float(train_acc),
                    'clean_train_acc': float(clean_train_acc), 'mem_acc': float(mem_acc),
                    'non_mem_acc': float(non_mem_acc)
                }
                if not args.debug:
                    wandb.log(train_metric)

                # test_stats
                test_acc, t_loss = eval_stu_model(student, test_loader)
                test_metric = {
                    'stu_test_acc': float(test_acc),
                    'stu_test_loss': float(np.mean(np.array(t_loss)))
                }
                if not args.debug:
                    wandb.log(test_metric)


def main():
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)

    # load cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                              download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                             download=True, transform=transform)

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # optional: subset cifar10 data
    targets = train_data.targets
    target_indices = np.arange(len(targets))
    train_1_idx, train_2_idx = train_test_split(target_indices, train_size=args.train_proportion, stratify=targets, random_state=1024)
    train_data_sub = Subset(train_data, train_1_idx)
    train_data_sub.targets = [train_data.targets[i] for i in train_1_idx]
    train_data_sub.data = [train_data.data[i] for i in train_1_idx]
    # down size test set size too
    targets = test_data.targets
    target_indices = np.arange(len(targets))
    test_1_idx, test_2_idx = train_test_split(target_indices, train_size=args.train_proportion, stratify=targets, random_state=1024)
    test_data_sub = Subset(test_data, test_1_idx)
    test_data_sub.targets = [test_data.targets[i] for i in test_1_idx]
    test_data_sub.data = [test_data.data[i] for i in test_1_idx]

    train_loader = torch.utils.data.DataLoader(train_data_sub, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # create canaries
    targets = train_data_sub.targets
    target_indices = np.arange(len(targets))
    train_idx, canary_idx = train_test_split(target_indices, train_size=(1-args.audit_proportion), stratify=targets, random_state=1024)
    canary_sub = Subset(train_data_sub, canary_idx)
    orig_targets = [train_data_sub.targets[i] for i in canary_idx]
    # idx = torch.randperm(torch.tensor(orig_targets).nelement())
    # new_targets = torch.tensor(orig_targets).view(-1)[idx].view(torch.tensor(orig_targets).size())
    # canary_sub.targets = new_targets
    canary_sub.targets = orig_targets
    canary_sub.data = [train_data_sub.data[i] for i in canary_idx]
    new_train_sub = Subset(train_data_sub, train_idx)
    mem_data, non_mem_data = torch.utils.data.random_split(
        canary_sub, [0.5, 0.5],
        generator=torch.Generator().manual_seed(1024))
    new_train_data = torch.utils.data.ConcatDataset([new_train_sub, mem_data])
    train_loader = torch.utils.data.DataLoader(new_train_data, batch_size=args.batch_size, shuffle=True)
    clean_train_loader = torch.utils.data.DataLoader(new_train_sub, batch_size=args.batch_size, shuffle=True)  # clean train
    mem_loader = torch.utils.data.DataLoader(mem_data, batch_size=args.batch_size, shuffle=True)  # noisy train
    non_mem_loader = torch.utils.data.DataLoader(non_mem_data, batch_size=args.batch_size, shuffle=True)  # noisy test
    if not args.debug:
        wandb.log({'num_mem': len(mem_data), 'num_non_mem': len(non_mem_data)})
    # mem_loader, non_mem_loader, clean_train_loader = train_loader, train_loader, train_loader

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.distill and args.non_priv:
        train(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)
    elif not args.distill and not args.non_priv:
        train_priv(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)
    elif args.distill and args.non_priv:
        teacher_model = load_model(args.load_path, args.load_exp_name, args.load_step)
        train_student(teacher_model, train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)
    elif args.distill and not args.non_priv:
        teacher_model = load_priv_model(args.load_path, args.load_exp_name, args.load_step)
        train_student(teacher_model, train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)


if __name__ == '__main__':
    main()
