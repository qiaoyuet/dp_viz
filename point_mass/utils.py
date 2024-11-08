import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle

from sim import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    save_path = os.path.join(save_path, exp_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'e{}.png'.format(epoch)))
    plt.close()


def save_model(model, step_num, model_path, exp_name):
    save_path = os.path.join(model_path, exp_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 's_{}.pt'.format(step_num)))


def load_model(load_path, device='cuda'):
    model = Net().to(device)
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    return model


def save_data_instance(data_dict, save_path):
    pickle.dump(data_dict, open(save_path, 'wb'))


def load_data_instance(load_path):
    data_dict = pickle.load(open(load_path, 'rb'))
    return data_dict
