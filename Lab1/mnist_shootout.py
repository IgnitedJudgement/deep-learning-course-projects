import os
import numpy as np
import torch
import torchvision
from matplotlib import image
from matplotlib import pyplot as plt

from Lab1 import data, pt_deep


def get_data():
    dataset_root = './data/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    return x_train, y_train, x_test, y_test


def pt_class_to_onehot(y_true):
    return torch.tensor(data.class_to_onehot(y_true))


def prepare_data(x_train, y_train, x_test, y_test):
    n_samples = x_train.shape[0]
    n_features = x_train.shape[1] * x_train.shape[2]
    n_classes = y_train.max().add_(1).item()

    x_tr = x_train.view(-1, n_features)
    y_tr = pt_class_to_onehot(y_train)
    x_te = x_test.view(-1, n_features)
    y_te = pt_class_to_onehot(y_test)

    return x_tr, y_tr, x_te, y_te


def data_to_gpu(x_train, y_train, x_test, y_test):
    return x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda()


if __name__ == "__main__":

    ### 1. Dio

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # x_train, y_train, x_test, y_test = prepare_data(*get_data())
    #
    # if device == 'cuda':
    #     x_train, y_train, x_test, y_test = data_to_gpu(x_train, y_train, x_test, y_test)
    #
    # param_niter, param_delta, param_lambdas = 1000, 0.1, [0, 0.01, 0.1, 1]
    #
    # for param_lambda in param_lambdas:
    #     path = f"data/task_1/lambda_{param_lambda}"
    #
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #
    #     model = pt_deep.Deep([784, 10], torch.relu).to(device)
    #     optimiser = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    #
    #     for iter in range(param_niter):
    #         loss = model.get_loss(x_train, y_train)
    #         loss.backward()
    #         optimiser.step()
    #         optimiser.zero_grad()
    #
    #         if iter % 10 == 0:
    #             print(f"Iteration: {iter}")
    #
    #     weights = model.weights[0].detach().cpu().numpy().T.reshape((-1, 28, 28))
    #
    #     for index, weight in enumerate(weights):
    #         image.imsave(os.path.join(path, f"{index}.png"), weight)

    ### 2. Dio

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # x_train, y_train, x_test, y_test = prepare_data(*get_data())
    #
    # if device == 'cuda':
    #     x_train, y_train, x_test, y_test = data_to_gpu(x_train, y_train, x_test, y_test)
    #
    # param_niter, param_delta, param_lambda = 2000, 0.1, 0.1
    #
    # arhitectures = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]
    #
    # loss_list = []
    #
    # for arhitecture in arhitectures:
    #     model = pt_deep.Deep(arhitecture, activation_function=torch.relu).to(device)
    #     optimiser = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    #
    #     arhitecture_loss_list = np.array([])
    #
    #     for iter in range(param_niter):
    #         loss = model.get_loss(x_train, y_train)
    #         loss.backward()
    #         optimiser.step()
    #         arhitecture_loss_list = np.append(arhitecture_loss_list, float(loss.detach().cpu().numpy()))
    #         optimiser.zero_grad()
    #
    #     loss_list.append(arhitecture_loss_list)
    #
    # for i in range(len(arhitectures)):
    #     plt.plot(np.arange(param_niter), loss_list[i], label=f"Arhitecture: {arhitecture[i]}")
    #     plt.legend()
    # plt.show()

    ### 3. Dio
    pass