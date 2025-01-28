import argparse
import wandb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import random
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import *
from auditing_utils import *

parser = argparse.ArgumentParser(description='pretrain')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--dataset', choices=['cifar'], default='cifar')
parser.add_argument('--train_proportion', default=0.1, type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--audit', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--data_path', default='/home/qiaoyuet/project/data', type=str)
parser.add_argument('--non_priv', action='store_true')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_mode', choices=['full', 'head'], default='head')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset(dataset):
    if dataset == 'cifar':
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        train_data = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                                  download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                                 download=True, transform=transform)
    else:
        raise NotImplementedError('Unknown dataset.')

    return train_data, test_data


@torch.no_grad()
def eval_model(net, loader, audit=False):
    net.eval()
    all_losses = []
    if audit:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # get per-sample loss for audit
    total, correct = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if audit:
            t_loss = criterion(outputs, labels)
            t_loss = torch_to_np(t_loss)
            all_losses.extend(t_loss)
    acc = correct / total
    return acc, all_losses


def train(train_loader, test_loader, **kwargs):
    if args.pretrain:
        net = models.resnet18(weights='IMAGENET1K_V1')
        if args.pretrain_mode == 'head':
            for param in net.parameters():
                param.requires_grad = False
            net.fc = torch.nn.Linear(512, 10)
        elif args.pretrain_mode == 'full':
            for param in net.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError
        net = net.to(device)
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params_to_update, lr=args.lr, weight_decay=0.001, momentum=0.9)
    else:
        net = models.resnet18(weights=None)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.001, momentum=0.9)

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
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
            step_counter += 1

            if step_counter % args.eval_every == 0:
                # train_stats
                train_acc, _ = eval_model(net, train_loader, audit=False)
                # clean_train_acc, _ = eval_model(net, clean_train_loader, audit=False)
                # mem_acc, _ = eval_model(net, mem_loader, audit=False)
                # non_mem_acc, _ = eval_model(net, non_mem_loader, audit=False)
                train_metric = {
                    'epoch': epoch, 'step': step_counter,
                    'train_loss': float(torch_to_np(loss)), 'train_acc': float(train_acc),
                    # 'clean_train_acc': float(clean_train_acc), 'mem_acc': float(mem_acc),
                    # 'non_mem_acc': float(non_mem_acc)
                }
                print(train_metric)

                # test_stats
                test_acc, t_loss = eval_model(net, test_loader, audit=False)
                test_metric = {
                    'test_acc': float(test_acc),
                    'test_loss': float(np.mean(np.array(t_loss)))
                }
                print(test_metric)

                if not args.debug:
                    metrics = {}
                    metrics.update(train_metric)
                    metrics.update(test_metric)
                    wandb.log(metrics)

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



def main():
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)

    train_data, test_data = load_dataset(args.dataset)

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train_proportion == 1:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    else:
        targets = train_data.targets
        target_indices = np.arange(len(targets))
        train_1_idx, train_2_idx = train_test_split(target_indices, train_size=args.train_proportion, stratify=targets, random_state=1024)
        train_data_sub = Subset(train_data, train_1_idx)
        train_data_sub.targets = [train_data.targets[i] for i in train_1_idx]
        train_data_sub.data = [train_data.data[i] for i in train_1_idx]
        train_loader = torch.utils.data.DataLoader(train_data_sub, batch_size=args.batch_size, shuffle=True)
        # # down size test set size too
        # targets = test_data.targets
        # target_indices = np.arange(len(targets))
        # test_1_idx, test_2_idx = train_test_split(target_indices, train_size=args.train_proportion, stratify=targets, random_state=1024)
        # test_data_sub = Subset(test_data, test_1_idx)
        # test_data_sub.targets = [test_data.targets[i] for i in test_1_idx]
        # test_data_sub.data = [test_data.data[i] for i in test_1_idx]
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        if args.audit:
            # create canaries
            targets = train_data_sub.targets
            target_indices = np.arange(len(targets))
            train_idx, canary_idx = train_test_split(target_indices, train_size=(1-args.audit_proportion), stratify=targets, random_state=1024)
            canary_sub = Subset(train_data_sub, canary_idx)
            orig_targets = [train_data_sub.targets[i] for i in canary_idx]
            # idx = torch.randperm(torch.tensor(orig_targets).nelement())
            # new_targets = torch.tensor(orig_targets).view(-1)[idx].view(torch.tensor(orig_targets).size())
            # canary_sub.targets = new_targets
            canary_sub.targets = orig_targets  # no label noise
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

    if args.non_private:
        train(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)
    else:
        train_priv(train_loader, test_loader, mem_loader, non_mem_loader, clean_train_loader)



if __name__ == '__main__':
    main()