import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import opacus
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import argparse
import wandb
import os
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--exp_proj', default='dp_viz', type=str)
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--valid_prop', default=0.2, type=float)
parser.add_argument("--non_priv", default=False, action='store_true')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--eval_every', default=100, type=int)
parser.add_argument("--dp_noise_multiplier", default=-1, type=float)
parser.add_argument("--dp_l2_norm_clip", default=1., type=float)
parser.add_argument("--virtual_batch_size", "--vb", type=int, metavar="VBACH_SIZE")
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--target_eps', type=float, default=None)
parser.add_argument("--out_path", type=str, default="/home/qiaoyuet/project/dp_viz/point_mass/outputs")
parser.add_argument("--save", default=False, action='store_true')
parser.add_argument("--plot", default=False, action='store_true')
parser.add_argument("--load_path", type=str, default=None)
args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self, init_dim, final_dim):
        super().__init__()
        self.linear1 = nn.Linear(init_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, final_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        return self.linear6(self.relu(self.linear5(self.relu(self.linear4(self.relu(self.linear3(
            self.relu(self.linear2(self.relu(self.linear1(x)))))))))))


def simulate_data(args):
    # multivariate gaussian
    f_0 = np.random.multivariate_normal(  # majority of class 0
        mean=[5, 5], cov=[[3, 0], [0, 3]], size=1000
    )
    f_1 = np.random.multivariate_normal(  # majority of class 1
        mean=[13, 5], cov=[[3, 0], [0, 3]], size=1000
    )
    # point mass gaussian
    f_0_1 = np.random.multivariate_normal(  # minority of class 1
        mean=[-3, 10], cov=[[1, 0], [0, 1]], size=50
    )
    f_1_1 = np.random.multivariate_normal(  # minority of class 0
        mean=[23, 0], cov=[[1, 0], [0, 1]], size=50
    )
    x1_0, x2_0 = np.concatenate([f_0[:, 0], f_1_1[:, 0]]), np.concatenate([f_0[:, 1], f_1_1[:, 1]])  # class 0
    x1_1, x2_1 = np.concatenate([f_1[:, 0], f_0_1[:, 0]]), np.concatenate([f_1[:, 1], f_0_1[:, 1]])  # class 1

    all_x1 = np.concatenate([x1_0, x1_1])
    all_x2 = np.concatenate([x2_0, x2_1])
    all_y = np.concatenate([np.zeros(len(x1_0)), np.ones(len(x1_1))]).astype(int)

    dat = pd.DataFrame({'x1': all_x1, 'x2': all_x2, 'y': all_y})

    x_data = torch.tensor(dat[['x1', 'x2']].values, dtype=torch.float32)
    y_data = torch.tensor(dat[['y']].values, dtype=torch.long)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, random_state=args.seed,
                                                          test_size=args.valid_prop, shuffle=True)
    return x_train, x_valid, y_train, y_valid


def np_to_torch(x, device='cuda'):
    return torch.from_numpy(x).to(torch.float32).to(device)


def torch_to_np(x):
    return x.cpu().detach().numpy()


def accuracy(preds, labels):
    return (torch_to_np(preds) == torch_to_np(labels)).mean()


def train_non_priv(args, x_train, x_valid, y_train, y_valid, device):
    model = MLP(init_dim=2, final_dim=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    x_train, x_valid = x_train.to(device), x_valid.to(device)
    y_train, y_valid = y_train.to(device), y_valid.to(device)

    step_counter = 0

    for epoch in tqdm(range(args.n_epochs + 1)):

        model.train()

        y_logits = model(x_train)
        loss = loss_fn(y_logits, y_train.flatten())
        _, y_pred = torch.max(y_logits, 1)
        acc = accuracy(y_pred, y_train.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_counter += 1

        if step_counter % args.eval_every == 0:
            model.eval()
            with torch.inference_mode():
                valid_logits = model(x_valid)
                valid_loss = loss_fn(valid_logits, y_valid.flatten())
                _, valid_pred = torch.max(valid_logits, 1)
                valid_acc = accuracy(valid_pred, y_valid.flatten())

            res_dict = {
                'Epoch': epoch, 'Step': step_counter, 'Train Loss': loss, 'Train Accuracy': acc,
                'Validation Loss': valid_loss, 'Validation Accuracy': valid_acc,
            }
            wandb.log(res_dict)

            if args.save:
                save_model(args, model, step_counter)


def train_priv(args, x_train, x_valid, y_train, y_valid, device):
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    valid_dataset = TensorDataset(x_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    model = MLP(init_dim=2, final_dim=2).to(device)
    model = ModuleValidator.fix(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    privacy_engine = PrivacyEngine(accountant='prv')

    if args.dp_noise_multiplier > -1:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=args.dp_l2_norm_clip,
            noise_multiplier=args.dp_noise_multiplier
        )
        wandb.log({'dp_noise_multiplier': args.dp_noise_multiplier})
    else:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.n_epochs,
            target_epsilon=args.target_eps,
            target_delta=args.delta,
            max_grad_norm=args.dp_l2_norm_clip,
        )
        wandb.log({'dp_noise_multiplier': optimizer.noise_multiplier})

    step_counter = 0
    for epoch in tqdm(range(args.n_epochs + 1)):

        model.train()

        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=len(train_dataset),
                optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (x_train, y_train) in enumerate(memory_safe_data_loader):
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                y_logits = model(x_train)
                loss = loss_fn(y_logits, y_train.flatten())
                _, y_pred = torch.max(y_logits, 1)
                acc = accuracy(y_pred, y_train.flatten())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_counter += 1

                if step_counter % args.eval_every == 0:
                    epsilon = privacy_engine.get_epsilon(args.delta)
                    model.eval()
                    with torch.inference_mode():
                        for x_valid, y_valid in valid_loader:
                            x_valid = x_valid.to(device)
                            y_valid = y_valid.to(device)

                            valid_logits = model(x_valid)
                            valid_loss = loss_fn(valid_logits, y_valid.flatten())
                            _, valid_pred = torch.max(valid_logits, 1)
                            valid_acc = accuracy(valid_pred, y_valid.flatten())

                    res_dict = {
                        'Epoch': epoch, 'Step': step_counter, 'Train Loss': loss, 'Train Accuracy': acc,
                        'Validation Loss': valid_loss, 'Validation Accuracy': valid_acc, 'Epsilon': epsilon
                    }
                    wandb.log(res_dict)

                    if args.save:
                        save_model(args, model, step_counter)


def save_model(args, model, step_num):
    save_path = os.path.join(args.out_path, args.exp_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 's_{}.pt'.format(step_num)))


def load_model(load_path):
    model = MLP(init_dim=2, final_dim=2)
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    return model


def load_model_priv(load_path, x_train, y_train):
    model = MLP(init_dim=2, final_dim=2)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    privacy_engine = PrivacyEngine(accountant='prv')
    model, _, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.dp_l2_norm_clip,
        noise_multiplier=args.dp_noise_multiplier
    )
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    return model


def infer(xinfer, model):
    model.eval()
    with torch.no_grad():
        ylogits = model(xinfer)
        _, yinfer = torch.max(ylogits, 1)
    return yinfer


def plot_decision_boundary(args, model, x_train, y_train, x_valid, y_valid, save_name, device='cuda'):
    model = model.to(device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    xlin1 = np.linspace(round(torch_to_np(x_train[:, 0]).min()) - 2, round(torch_to_np(x_train[:, 0]).max()) + 2, 500)
    xlin2 = np.linspace(round(torch_to_np(x_train[:, 1]).min()) - 2, round(torch_to_np(x_train[:, 1]).max()) + 2, 500)
    xx1, xx2 = np.meshgrid(xlin1, xlin2)
    xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
    xinfer = np_to_torch(xinfer)
    yinfer = infer(xinfer, model)
    yinfer = torch_to_np(yinfer)
    yy = np.reshape(yinfer, xx1.shape)

    # make contour of decision boundary
    ax1.contourf(xx1, xx2, yy, alpha=.5, cmap='rainbow')

    # plot class 0
    # correctly/wrongly predicted is plotted as point/cross
    xinfer = x_train[y_train.ravel() == 0]
    xinfer = xinfer.to(device)
    yinfer = infer(xinfer, model)
    xinfer = torch_to_np(xinfer)
    yinfer = torch_to_np(yinfer)
    ax1.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], '.',
             color='blue', markersize=8, label='class 0')
    ax1.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], 'x',
             color='blue', markersize=8, label='class 0 error')

    # plot class 1
    # correctly/wrongly predicted is plotted as point/cross
    xinfer = x_train[y_train.ravel() == 1]
    xinfer = xinfer.to(device)
    yinfer = infer(xinfer, model)
    xinfer = torch_to_np(xinfer)
    yinfer = torch_to_np(yinfer)
    ax1.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], '.',
             color='red', markersize=8, label='class 1')
    ax1.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], 'x',
             color='red', markersize=8, label='class 1 error')

    ax1.set_title('train')
    # ax1.legend(loc='lower left', framealpha=.5, fontsize=10)

    ## priv model
    xlin1 = np.linspace(round(torch_to_np(x_valid[:, 0]).min()) - 2, round(torch_to_np(x_valid[:, 0]).max()) + 2, 500)
    xlin2 = np.linspace(round(torch_to_np(x_valid[:, 1]).min()) - 2, round(torch_to_np(x_valid[:, 1]).max()) + 2, 500)
    xx1, xx2 = np.meshgrid(xlin1, xlin2)
    xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
    xinfer = np_to_torch(xinfer)
    yinfer = infer(xinfer, model)
    yinfer = torch_to_np(yinfer)
    yy = np.reshape(yinfer, xx1.shape)

    # make contour of decision boundary
    ax2.contourf(xx1, xx2, yy, alpha=.5, cmap='rainbow')

    # plot class 0
    # correctly/wrongly predicted is plotted as point/cross
    xinfer = x_valid[y_valid.ravel() == 0]
    xinfer = xinfer.to(device)
    yinfer = infer(xinfer, model)
    xinfer = torch_to_np(xinfer)
    yinfer = torch_to_np(yinfer)
    ax2.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], '.',
             color='blue', markersize=8, label='class 0')
    ax2.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], 'x',
             color='blue', markersize=8, label='class 0 error')

    # plot class 1
    # correctly/wrongly predicted is plotted as point/cross
    xinfer = x_valid[y_valid.ravel() == 1]
    xinfer = xinfer.to(device)
    yinfer = infer(xinfer, model)
    xinfer = torch_to_np(xinfer)
    yinfer = torch_to_np(yinfer)
    ax2.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], '.',
             color='red', markersize=8, label='class 1')
    ax2.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], 'x',
             color='red', markersize=8, label='class 1 error')

    ax2.set_title('test')
    # ax2.legend(loc='lower left', framealpha=.5, fontsize=10)

    # plt.show()
    save_path = os.path.join(args.load_path, 'img')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    fig.savefig(os.path.join(save_path, '{}.png'.format(save_name)))
    plt.close()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train, x_valid, y_train, y_valid = simulate_data(args)

    if not args.plot:
        # train
        if not args.debug:
            wandb.init(project=args.exp_proj, config=args, group=args.exp_group, name=args.exp_name)

        if args.non_priv:
            train_non_priv(args, x_train, x_valid, y_train, y_valid, device)
        else:
            train_priv(args, x_train, x_valid, y_train, y_valid, device)
    else:
        # plot
        ckpts = os.listdir(args.load_path)
        for ckpt in ckpts:
            if ckpt.endswith('.pt'):
                if args.non_priv:
                    tmp_model = load_model(os.path.join(args.load_path, ckpt))
                else:
                    tmp_model = load_model_priv(os.path.join(args.load_path, ckpt), x_train, y_train)
                plot_decision_boundary(args, tmp_model, x_train, y_train, x_valid, y_valid, str(ckpt).split('.')[0])


if __name__ == '__main__':
    main(args)
