import argparse
import numpy as np
import torch
import math
import csv
import random
import scipy
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from utils_qt import myMultiStepLR
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
parser.add_argument("--non_private", default=False, action='store_true')
parser.add_argument("--dp_noise_multiplier", default=1., type=float)
parser.add_argument("--dp_l2_norm_clip", default=1., type=float)
parser.add_argument('--save', action='store_true')
parser.add_argument("--scale", default=1.5, type=float)
# parser.add_argument("--log_output_file", type=str, default="tmp.csv") # new arg for O(1) logging
            
def bernoulli_sample_datasets(single_dataset, p=0.5):
    """
    Sample datasets using Bernoulli trials to determine inclusion in the member or non-member dataset.
    
    Supports both PyTorch datasets and NumPy arrays.
    
    Args:
        single_dataset (Dataset or tuple): The original dataset from which to sample.
                                           If a tuple, it should be (data, targets) where both are NumPy arrays.
        p (float, optional): Probability of a data point being included in the member dataset. Default is 0.5.

    Returns:
        tuple: (sampled_member_dataset, sampled_non_member_dataset) where:
            - sampled_member_dataset (Subset or tuple): Subset of the original dataset where data points were included based on Bernoulli trials.
            - sampled_non_member_dataset (Subset or tuple): Subset of the original dataset where data points were excluded based on Bernoulli trials.
    """
    
    # Handle NumPy arrays by converting them to tensors
    if isinstance(single_dataset, tuple):
        data, targets = single_dataset
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        single_dataset = torch.utils.data.TensorDataset(data, targets)
    
    # Total number of data points in the original dataset
    total_length = len(single_dataset)
    
    # Create Bernoulli trials for each data point
    bernoulli_trials = torch.bernoulli(torch.full((total_length,), p))

    # Separate indices based on Bernoulli trials
    member_indices = [i for i in range(total_length) if bernoulli_trials[i]]
    non_member_indices = [i for i in range(total_length) if not bernoulli_trials[i]]
    
    # Create subsets based on the indices
    sampled_member_dataset = Subset(single_dataset, member_indices)
    sampled_non_member_dataset = Subset(single_dataset, non_member_indices)
    
    return sampled_member_dataset, sampled_non_member_dataset
    
def p_value_DP_audit(m, r, v, eps, delta=0):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        eps,delta = DP guarantee of null hypothesis
    Returns:
        p-value = probability of >=v correct guesses under null hypothesis
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
       sum = sum + scipy.stats.binom.pmf(v - i, r, q)
       if sum > i * alpha:
           alpha = sum / i
    p = beta  #+ alpha * delta * 2 * m
    # print("p", p)
    return min(p, 1)

def get_eps_audit(m, r, v,  p, delta):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        p = 1-confidence e.g. p=0.05 corresponds to 95%
    Returns:
        lower bound on eps i.e. algorithm is not (eps,delta)-DP
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p <= 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 1 # maintain p_value_DP(eps_max) >= p

    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min

def find_O1_pred(member_loss_values, non_member_loss_values, delta = 0.):
    """
    Args:
        member_loss_values: NumPy array containing member loss values
        non_member_loss_values: NumPy array containing non_member loss values
    Returns:
     best_eps: largest audit (epsilon) value that can be returned for a particular p value
    """
    
    # Create labels for real and generated loss values
    member_labels = np.ones_like(member_loss_values)
    non_member_labels = np.zeros_like(non_member_loss_values)

    # Concatenate loss values and labels
    all_losses = np.concatenate((member_loss_values, non_member_loss_values))
    all_labels = np.concatenate((member_labels, non_member_labels))
    
    # Step 1: Find t_pos that maximizes precision for positive predictions
    best_precision = 0
    best_t_pos = 0
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
    results, recall = [], []
    best_accuracy = 0
    best_t_neg = 0
    total_predictions = 0
    correct_predictions = 0
    best_eps = 0
    p = 0.05
    for t_pos in threshold_range:
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)
        
        eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
        precision = true_positives / len(positive_predictions)
        if eps > best_eps:
            print("EPSILON UPDATE:", eps)
            best_eps = eps
            best_t_pos = t_pos
        recalls = true_positives / np.sum(all_labels == 1)
        recall.append(recalls)
        
        # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
        for t_neg in reversed(threshold_range):
            if t_neg <= best_t_pos:
                break
            confident_predictions = all_losses[(all_losses <= best_t_pos) | (all_losses >= t_neg)]
            r = len(confident_predictions)
            mask_pos = (confident_predictions <= best_t_pos) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 1)
            mask_neg = (confident_predictions >= t_neg) & (all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 0)

            v = np.sum(np.logical_or(mask_pos, mask_neg))

            if r > 0:
                accuracy = v / r
                eps = get_eps_audit(len(all_labels), r, v, p, delta)
                if eps > best_eps:
                    best_eps = eps
                    best_t_neg = t_neg
                    total_predictions = r
                    correct_predictions = v
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
            
            results.append({
                't_pos': t_pos,
                't_neg': best_t_neg,
                'best_precision': precision,
                'best_accuracy': best_accuracy,
                'recall': recall,
                'total_predictions': r,
                'correct_predictions': v
            })
    print(f"Best eps: {best_eps} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
    print(f"Best precision for t_pos: {best_precision} with t_pos: {best_t_pos}")
    print(f"Best accuracy: {best_accuracy} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
    
    # Save results to CSV file
    output_csv_path = "swiss_audit_over.csv"
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['t_pos', 't_neg', 'best_precision', 'best_accuracy', 'recall', 'total_predictions', 'correct_predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return total_predictions, correct_predictions, len(all_losses)
    
class CustomDataset(Dataset):
    """
    Custom dataset class to hold data and labels.

    Attributes:
        data (list or tensor): The data samples.
        targets (list or tensor): The corresponding labels for the data samples.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def insert_canaries(dataset, target_label, new_label, num_canaries):
    """
    Insert canary samples into the dataset by changing the labels of some data points.

    Args:
        dataset (CustomDataset): The original dataset.
        target_label (int): The original label of the canary samples.
        new_label (int): The new label to assign to the canary samples.
        num_canaries (int): The number of canary samples to insert.

    Returns:
        CustomDataset: The modified dataset with canary samples inserted.
    """
    # Find all indices of the target_label
    target_indices = [i for i, label in enumerate(dataset.targets) if label == target_label]

    # Ensure num_canaries does not exceed the number of available target_label samples
    assert num_canaries <= len(target_indices), "Number of canaries to insert exceeds available data with target_label"

    # Randomly select num_canaries from the target_indices
    canary_indices = random.sample(target_indices, num_canaries)

    # Create new data and targets lists
    new_data = list(dataset.data)
    new_targets = list(dataset.targets)

    # Assign new labels to the selected canary indices
    for idx in canary_indices:
        new_targets[idx] = new_label

    # Return the modified dataset
    return CustomDataset(new_data, new_targets)
    
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
        self.train_loader, self.test_loader, self.model, self.criterion, self.optimizer, self.scheduler, self.non_member_loader = self.make_dataset_and_model()

    @staticmethod
    def make_dataset_and_model():
        X, y = twospirals(args.n_data // 2, noise=args.noise, scale=args.scale)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, stratify=y)
        train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        if args.batch_size is None: args.batch_size = args.n_data; print('fullbatch gradient descent')
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        train_set, non_member_set = bernoulli_sample_datasets(train_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=args.batch_size, shuffle=False)

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
        return train_loader, test_loader, model, criterion, optimizer, scheduler, non_member_loader
    
    
    
    def train(self):
        self.model.train()
        k = 10
        
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
                self.eval(self.test_loader, get_train_acc=True)

            if args.plot_every > 0 and self.step_counter % args.plot_every == 0:
                model.plot(name='plot_' + str(self.step_counter) + '.jpg', plttitle='plot_' + str(self.step_counter))
                member_loss_values = self.eval(self.train_loader)
                print("STEPS:", self.step_counter)
                non_member_loss_values = self.eval(self.non_member_loader)
                find_O1_pred(member_loss_values, non_member_loss_values)

            # TODO: for every k steps, return O(1) calculation
            # we need non-member data -> O(1) tests on both positive class and negative class
            # we can have accuracy in this case, maybe it should be easier given the metrics we choose later
            # sampling of member and non-member data needs to be done via Bernoulli trials for independent sampling
            # if epoch % k == 0:
            #     member_loss_values = self.eval(self.train_loader)
            #     non_member_loss_values = self.eval(self.non_member_loader)
            #     find_O1_pred(member_loss_values, non_member_loss_values)
            

    def eval(self, test_loader, get_train_acc=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        test_array = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)
                y_pred = self.model(X)
                loss = self.criterion(y_pred.squeeze(), y)
                test_array.append(loss.item())
                test_loss += loss.item()
                # _, predicted = y_pred.max(1)
                predicted = F.sigmoid(y_pred).squeeze()
                predicted[predicted > .5] = 1
                predicted[predicted <= .5] = 0
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            test_L = test_loss / (len(test_loader))
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
        return test_array

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
            raise NotImplementedError  # fixme

        if len(self.test_loader) == 1:
            xtest = next(iter(self.test_loader))[0].numpy()
            ytest = next(iter(self.test_loader))[1].numpy()
        else:
            raise NotImplementedError  # fixme

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
