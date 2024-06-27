import argparse
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from utils import myMultiStepLR
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, \
    figure, close, suptitle, tight_layout, contourf, xlim, ylim
import os
from os.path import join, basename, dirname, exists
import pickle
from time import time, sleep
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.utils.prune as prune

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--n_data', default=400, type=int)
parser.add_argument('--noise', default=1, type=float, help="data generation noise")
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--n_hidden', default=[23, 16, 26, 32, 28, 31], type=int, nargs='+')
parser.add_argument('--n_dim', default=2, type=int)
parser.add_argument('--n_class', default=1, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--n_epoch', default=40000, type=int)
parser.add_argument("--exp_group", type=str, default="tmp")
parser.add_argument("--exp_name", type=str, default="tmp")
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--eval_every", default=1, type=int)
parser.add_argument("--plot_every", default=-1, type=int)
parser.add_argument("--compress_at", default=-1, type=int)
parser.add_argument("--compress_amt", default=0.2, type=float)
parser.add_argument("--non_private", default=False, action='store_true')
parser.add_argument("--dp_noise_multiplier", default=1., type=float)
parser.add_argument("--dp_l2_norm_clip", default=1., type=float)
parser.add_argument('--save', action='store_true')
parser.add_argument("--scale", default=1.5, type=float)


def twospirals(n_points, noise=.5, scale=1.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 600 * (2 * np.pi) / 360
    d1x = -scale * np.cos(n) * n + np.random.randn(n_points, 1) * noise
    d1y = scale * np.sin(n) * n + np.random.randn(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.input_dim = self.args.n_dim
        self.output_dim = self.args.n_class
        self.hidden_dim = self.args.n_hidden
        current_dim = self.input_dim
        self.layers = nn.ModuleList()
        for hdim in self.hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # x = F.softmax(self.layers[-1](x))
        x = self.layers[-1](x)
        return x


class SwissRoll:
    def __init__(self, args):
        super(SwissRoll, self).__init__()
        self.args = args
        self.step_counter = 0
        self.train_loader, self.test_loader, self.model, self.criterion, self.optimizer, self.scheduler = self.make_dataset_and_model()

    @staticmethod
    def make_dataset_and_model():
        X, y = twospirals(args.n_data // 2, noise=args.noise, scale=args.scale)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, stratify=y)
        train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        if args.batch_size is None: args.batch_size = args.n_data; print('fullbatch gradient descent')
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = MLP(args).to(device)
        if not args.non_private:
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(errors[-1])
                model = ModuleValidator.fix(model)  # batch norm -> group norm

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        # scheduler = MultiStepLR(optimizer, milestones=[3000, 30000, 300000], gamma=0.1)
        scheduler = myMultiStepLR(optimizer, milestones=[3062, 28000, 300000], gamma=[0.1, 0.01, 0.001])
        if not args.non_private:
            privacy_engine = PrivacyEngine(accountant='prv')
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.dp_noise_multiplier,
                max_grad_norm=args.dp_l2_norm_clip,
            )
        return train_loader, test_loader, model, criterion, optimizer, scheduler

    def train(self):
        self.model.train()
        for epoch in tqdm(range(args.n_epoch)):
            losses = 0
            for i, (X, y) in enumerate(self.train_loader):
                X = X.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.criterion(y_pred.squeeze(), y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.step_counter += 1
                losses += loss.item()
            if epoch % args.eval_every == 0:
                mean_loss = losses / len(self.train_loader)
                print('Epoch %d | Step %d | Loss %6.2f' % (epoch, self.step_counter, mean_loss))
                if not self.args.debug:
                    wandb.log({'Epoch': epoch, 'Step': self.step_counter,
                               'Train Loss': mean_loss, 'Cur_lr': float(self.scheduler.get_last_lr()[0])})
                self.eval(get_train_acc=True)

            if epoch == args.compress_at:
                self.compress(self.args.compress_amt)
                if not args.debug:
                    global_sparsity = 100. * float(
                        torch.sum(self.model.layers[1].weight == 0)
                        + torch.sum(self.model.layers[2].weight == 0)
                        # + torch.sum(self.model.layers[3].weight == 0)
                        # + torch.sum(self.model.layers[4].weight == 0)
                        # + torch.sum(self.model.layers[5].weight == 0)
                    ) / float(
                        self.model.layers[1].weight.nelement()
                        + self.model.layers[2].weight.nelement()
                        # + self.model.layers[3].weight.nelement()
                        # + self.model.layers[4].weight.nelement()
                        # + self.model.layers[5].weight.nelement()
                    )
                    wandb.log({'global_sparsity': global_sparsity})

            if args.plot_every > 0 and self.step_counter % args.plot_every == 0:
                model.plot(name='plot_' + str(self.step_counter) + '.jpg', plttitle='plot_' + str(self.step_counter))

            # TODO: for every k steps, return O(1) calculation

    def eval(self, get_train_acc=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
                y_pred = self.model(X)
                loss = self.criterion(y_pred.squeeze(), y)
                test_loss += loss.item()
                # _, predicted = y_pred.max(1)
                predicted = F.sigmoid(y_pred).squeeze()
                predicted[predicted > .5] = 1
                predicted[predicted <= .5] = 0
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            test_L = test_loss / (len(self.test_loader))
        test_acc = 100. * correct / total
        if not args.debug:
            wandb.log({'Test_step': self.step_counter, 'Test_loss': test_L, 'Test_acc': test_acc})

        if get_train_acc:
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in self.train_loader:
                    X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
                    y_pred = self.model(X)
                    predicted = F.sigmoid(y_pred).squeeze()
                    predicted[predicted > .5] = 1
                    predicted[predicted <= .5] = 0
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
            train_acc = 100. * correct / total
            if not args.debug:
                wandb.log({'Train_acc': train_acc})
        else:
            train_acc = np.nan
        print('Step %d | Train_acc %6.2f | Test_acc %6.2f' % (self.step_counter, train_acc, test_acc))

    def compress(self, perc):
        # random global pruning
        parameters_to_prune = (
            (self.model.layers[1], 'weight'),
            (self.model.layers[2], 'weight'),
            # (self.model.layers[3], 'weight'),
            # (self.model.layers[4], 'weight'),
            # (self.model.layers[5], 'weight'),
        )
        prune.global_unstructured(
            parameters_to_prune,
            # pruning_method=prune.RandomUnstructured,
            pruning_method=prune.L1Unstructured,
            amount=perc,
        )
        for layer_num in list(range(1, len(self.model.layers)-1)):
            prune.remove(self.model.layers[layer_num], 'weight')

    def infer(self, xinfer):
        '''inference on a batch of input data xinfer. outputs collapsed to 1 or 0'''
        self.model.eval()
        with torch.no_grad():
            xinfer = torch.from_numpy(xinfer).to(torch.float32).to(device)
            ypred = self.model(xinfer)
            yinfer = F.sigmoid(ypred).squeeze()
            yinfer[yinfer > .5] = 1
            yinfer[yinfer <= .5] = 0
        return yinfer.cpu().detach().numpy()

    def plot(self, name='plot.jpg', plttitle='plot', index=0):
        '''plot decision boundary alongside loss surface'''
        if len(self.train_loader) == 1:
            xtrain = next(iter(self.train_loader))[0].numpy()
            ytrain = next(iter(self.train_loader))[1].numpy()
        else:
            raise NotImplementedError

        if len(self.test_loader) == 1:
            xtest = next(iter(self.test_loader))[0].numpy()
            ytest = next(iter(self.test_loader))[1].numpy()
        else:
            raise NotImplementedError

        # make contour of decision boundary
        xlin = (round(xtrain.max()) + 1) * np.linspace(-1, 1, 500)
        xx1, xx2 = np.meshgrid(xlin, xlin)
        xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
        yinfer = self.infer(xinfer)
        yy = np.reshape(yinfer, xx1.shape)

        # define plot class
        # figure(figsize=(38, 6))
        # plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # training set
        # plot the decision boundary
        ax1.contourf(xx1, xx2, yy, alpha=.5, cmap='rainbow')

        # plot blue class
        # class 0, correctly/wrongly predicted is plotted as point/cross
        xinfer = xtrain[ytrain.ravel() == 0]
        yinfer = self.infer(xinfer)
        ax1.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], '.', color=[0, 0, .5], markersize=8,
             label='class 1')
        ax1.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], 'x', color=[0, 0, .5], markersize=8,
             label='class 1 error')
        xinferblue, yinferblue = xinfer, yinfer

        # plot red class
        # class 1, correctly/wrongly predicted is plotted as point/cross
        xinfer = xtrain[ytrain.ravel() == 1]
        yinfer = self.infer(xinfer)
        ax1.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], '.', color=[.5, 0, 0], markersize=8,
             label='class 2')
        ax1.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], 'x', color=[.5, 0, 0], markersize=8,
             label='class 2 error')
        xinferred, yinferred = xinfer, yinfer

        # # plot training data only
        # class_0 = xtrain[ytrain.ravel() == 0]
        # plot(class_0[:, 0], class_0[:, 1], '.', color=[0, 0, .5], markersize=8, label='class 0')
        # class_1 = xtrain[ytrain.ravel() == 1]
        # plot(class_1[:, 0], class_1[:, 1], '.', color=[.5, 0, 0], markersize=8, label='class 1')

        ax1.set_title('train')
        ax1.legend(loc='lower left', framealpha=.5, fontsize=10)

        # test set
        # plot the decision boundary
        ax2.contourf(xx1, xx2, yy, alpha=.5, cmap='rainbow')

        # plot blue class
        # class 0, correctly/wrongly predicted is plotted as point/cross
        xinfer = xtest[ytest.ravel() == 0]
        yinfer = self.infer(xinfer)
        ax2.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], '.', color=[0, 0, .5], markersize=8,
             label='class 1')
        ax2.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], 'x', color=[0, 0, .5], markersize=8,
             label='class 1 error')
        xinferblue, yinferblue = xinfer, yinfer

        # plot red class
        # class 1, correctly/wrongly predicted is plotted as point/cross
        xinfer = xtest[ytest.ravel() == 1]
        yinfer = self.infer(xinfer)
        ax2.plot(xinfer[yinfer.ravel() == 1, 0], xinfer[yinfer.ravel() == 1, 1], '.', color=[.5, 0, 0], markersize=8,
             label='class 2')
        ax2.plot(xinfer[yinfer.ravel() == 0, 0], xinfer[yinfer.ravel() == 0, 1], 'x', color=[.5, 0, 0], markersize=8,
             label='class 2 error')
        xinferred, yinferred = xinfer, yinfer

        ax2.set_title('test')
        ax2.legend(loc='lower left', framealpha=.5, fontsize=10)

        # load data from surface plots
        if exists(join(logdir, 'surface.pkl')):
            with open(join(logdir, 'surface.pkl'), 'rb') as f:
                cfeed, xent, acc, spec = pickle.load(f)

            # surface of xent
            plt.subplot2grid((3, 4), (0, 0))
            plot(cfeed, xent, '-', color='orange')
            plot(cfeed[index], xent[index], 'ko', markersize=8)
            title('xent')
            ylim(0, 20)
            plt.gca().axes.get_xaxis().set_ticklabels([])

            # surface of acc
            plt.subplot2grid((3, 4), (1, 0))
            plot(cfeed, acc, '-', color='green')
            plot(cfeed[index], acc[index], 'ko', markersize=8)
            title('acc')
            ylim(0, 1.05)
            plt.gca().axes.get_xaxis().set_ticklabels([])

            # surface of spec
            plt.subplot2grid((3, 4), (2, 0))
            plot(cfeed, spec, '-', color='cyan')
            plot(cfeed[index], spec[index], 'ko', markersize=8)
            title('curv')
            ylim(0, 1700000)

        suptitle(plttitle)
        tight_layout()

        # image metadata and save image
        os.makedirs(join(logdir, args.exp_name), exist_ok=True)
        savefig(join(logdir, args.exp_name, name))
        # if name=='plot.jpg': experiment.log_image(join(logdir, 'images/plot.jpg')); os.remove(join(logdir, 'images/plot.jpg'))
        sleep(.1)
        close('all')

        # save the data needed to reproduce plot
        if args.save:
            with open(join(logdir, 'plotdata.pkl'), 'wb') as f:
                pickle.dump(dict(xinferred=xinferred, yinferred=yinferred, xinferblue=xinferblue, yinferblue=yinferblue,
                                 xx1=xx1, xx2=xx2, yy=yy, xtrain=xtrain, ytrain=ytrain), f)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.debug:
        wandb.login(key='b7617eecafac1c7019d5cf07b1aadac73891e3d8')
        wandb.init(project='dp_viz', config=vars(args), group=args.exp_group, name=args.exp_name)
    logdir = './output/'
    os.makedirs(logdir, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SwissRoll(args)
    model.train()
    model.plot(name='plot_' + str(args.exp_name) + '.jpg')
