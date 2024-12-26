import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb
import numpy as np
import random

from utils import load_model, load_data_instance, np_to_torch, torch_to_np, save_plot, load_priv_model

parser = argparse.ArgumentParser(description='DistillSim')
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epoch_stu', default=1000, type=int)
parser.add_argument('--eval_every_stu', default=1, type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--exp_group', default='tmp', type=str)
parser.add_argument('--exp_name', default='tmp', type=str)
parser.add_argument('--load_path', default='/home/qiaoyuet/project/dp_viz/point_mass/outputs/sim', type=str)
parser.add_argument('--load_exp_name', default='tmp', type=str)
parser.add_argument('--load_step', default=-1, type=int)
parser.add_argument('--alpha', default=0.1, type=float, help='distillation loss strength')
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--non_priv', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StudentNet(torch.nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 32)
        # self.hidden2 = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        # x = torch.relu(self.hidden2(x))
        # x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


# Distillation loss function for regression
def distillation_loss(student_outputs, teacher_outputs, true_labels, alpha):
    ground_truth_loss = F.mse_loss(student_outputs, true_labels)
    distill_loss = F.mse_loss(student_outputs, teacher_outputs)
    return alpha * ground_truth_loss + (1 - alpha) * distill_loss


@torch.no_grad()
def eval_student(net, test_x, test_y, criterion):
    net.eval()
    y_pred = net(test_x.unsqueeze(1))
    t_loss = criterion(y_pred.squeeze(), test_y)
    t_loss = torch_to_np(t_loss)
    y_pred = torch_to_np(y_pred)
    return y_pred, t_loss


# Training loop for student model
def train_student(teacher, data_dict):

    student = StudentNet().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr)
    criterion_stu = torch.nn.MSELoss()

    x = np_to_torch(data_dict.get('train_x'))
    y = np_to_torch(data_dict.get('train_y'))
    test_x = np_to_torch(data_dict.get('test_x'))
    test_y = np_to_torch(data_dict.get('test_y'))

    for epoch in range(args.n_epoch_stu):
        teacher.eval()
        student.train()
        optimizer.zero_grad()
        outputs = student(x.unsqueeze(1))
        with torch.no_grad():
            soft_outputs = teacher(x.unsqueeze(1))
        loss = distillation_loss(outputs.squeeze(), soft_outputs.squeeze(), y, args.alpha)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every_stu == 0:
            # train_stats
            train_metric = {
                'epoch': epoch,
                'train_loss': float(torch_to_np(loss))
            }
            if not args.debug:
                wandb.log(train_metric)

            # test_stats
            y_pred, t_loss = eval_student(student, test_x, test_y, criterion_stu)
            test_metric = {
                'test_loss': float(t_loss)
            }
            if not args.debug:
                wandb.log(test_metric)

            # save_plot
            if not args.debug and not args.no_plot:
                save_plot(x, y, student, epoch, args.save_path, args.exp_name)


def distill():
    if not args.debug:
        wandb.login()
        run = wandb.init(project="dp_viz", group=args.exp_group, name=args.exp_name)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data instance
    data_path = os.path.join(args.load_path, args.load_exp_name, 'data_instance.pkl')
    data_dict = load_data_instance(data_path)

    # load teacher model
    if args.non_priv:
        teacher_model = load_model(args.load_path, args.load_exp_name, args.load_step)
    else:
        teacher_model = load_priv_model(args.load_path, args.load_exp_name, args.load_step)

    # train student model
    train_student(teacher_model, data_dict)


if __name__ == '__main__':
    distill()
