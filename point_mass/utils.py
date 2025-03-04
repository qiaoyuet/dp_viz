import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import os
import pickle
from opacus.validators import ModuleValidator
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, __value):
        return setattr(self.dataset, 'targets', __value)

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, __value):
        return setattr(self.dataset, 'transform', __value)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class CanariesDataset(WrapperDataset):
    def __init__(self, dataset, num_canaries=1):
        super().__init__(dataset)

        n_labels = len(dataset.dataset.classes)
        assert num_canaries > 0
        labels = np.array(dataset.dataset.targets[dataset.indices])
        mask = np.arange(0, len(labels)) < num_canaries
        np.random.seed(1024)
        np.random.shuffle(mask)
        rnd_labels = np.random.choice(n_labels, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.noisy_targets = labels

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        y = self.noisy_targets[i]
        return X, y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64)
        self.hidden2 = torch.nn.Linear(64, 128)
        self.hidden3 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


class CNNSmall(torch.nn.Module):
    def __init__(self):
        super(CNNSmall, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CNNCifar(torch.nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 10)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class StudentCNN(torch.nn.Module):
    def __init__(self, out_channels, num_classes):
        super(StudentCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, out_channels, kernel_size=5)
        self.fc1 = torch.nn.Linear(out_channels * 12 * 12, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class StudentNet(torch.nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, num_classes):
        super(StudentNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, num_classes)
        self.num_hidden = num_hidden

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        if self.num_hidden > 0:
            for _ in range(self.num_hidden):
                out = self.fc2(out)
                out = self.relu(out)
        out = self.fc3(out)
        return out


def np_to_torch(x):
    return torch.from_numpy(x).to(torch.float32).to(device)
    # return torch.from_numpy(x).to(torch.float32)


def torch_to_np(x):
    return x.cpu().detach().numpy()


def save_plot(train_x, train_y, net, epoch, save_path, exp_name):
    # plot true function
    f = lambda x: np.sin(3 * x)
    x_plot = np.linspace(0, 2, 100)
    actual_y = [f(p).item() for p in x_plot]
    plt.plot(x_plot, actual_y, 'g', label='Actual Function')

    # plot train data
    plt.scatter(train_x, train_y)

    # plot est function
    x_plot = np.linspace(0, 2, 100)
    net.eval()
    with torch.no_grad():
        predicted_y = net(np_to_torch(x_plot).unsqueeze(1)).squeeze()
        predicted_y = torch_to_np(predicted_y)
    plt.plot(x_plot, predicted_y, 'b', label='Predicted Function')
    plt.legend()
    # plt.show()
    save_path = os.path.join(save_path, exp_name, 'img')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'e_{}.png'.format(epoch)))
    plt.close()


def save_model(model, step_num, model_path, exp_name):
    save_path = os.path.join(model_path, exp_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    os.makedirs(os.path.join(model_path, exp_name, 'ckpt'), exist_ok=True)
    save_path2 = os.path.join(model_path, exp_name, 'ckpt')
    torch.save(model.state_dict(), os.path.join(save_path2, 's_{}.pt'.format(step_num)))


def load_model(load_path, model, exp_name, load_step, device='cuda'):
    model_path = os.path.join(load_path, exp_name, 'ckpt', 's_{}.pt'.format(load_step))
    # model = Net().to(device)
    # model = CNNSmall().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()
    return model


def load_priv_model(load_path, model, exp_name, load_step, device='cuda'):
    model_path = os.path.join(load_path, exp_name, 'ckpt', 's_{}.pt'.format(load_step))
    # model = Net().to(device)
    # model = CNNSmall().to(device)
    # model = ModuleValidator.fix(model)

    # priv engine changes module names, needs to change back when loading
    loaded_model = torch.load(model_path, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in loaded_model.items():
        assert '_module.' in k
        if k[:8] == '_module.':
            name = k[8:]  # remove `_module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.eval()
    return model


def save_data_instance(data_dict, save_path):
    pickle.dump(data_dict, open(save_path, 'wb'))


def load_data_instance(load_path):
    data_dict = pickle.load(open(load_path, 'rb'))
    return data_dict
