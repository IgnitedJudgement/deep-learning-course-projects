import os

import skimage as ski
from skimage import img_as_ubyte
import skimage.io

import math
import numpy as np

import torch

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def draw_conv_filters(epoch, step, layer, name, save_dir):
    w = layer.weight.clone().detach().numpy()
    N, C, H, W = w.shape[:4]

    num_filters = N
    k = H

    w -= w.min()
    w /= w.max()

    border, cols = 1, 8
    rows = math.ceil(num_filters / cols)

    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border

    for i in range(1):
        img = np.zeros([height, width])

        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r + k, c:c + k] = w[j, i]

        filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (name, epoch, step, i)
        ski.io.imsave(os.path.join(save_dir, filename), img_as_ubyte(img))


def draw_conv_filters_color(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img_as_ubyte(img))


def accuracy_score(outputs, labels):
    predictions = torch.max(outputs, dim=1)[1]

    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))


def validation_step(model, batch):
    features, labels = batch
    predictions = model.forward(features)

    loss = F.cross_entropy(predictions, labels)
    accuracy = accuracy_score(predictions, labels)

    return {'validation_loss': loss, 'validation_accuracy': accuracy}


def validation_epoch_end(outputs):
    epoch_loss = torch.stack([output['validation_loss'] for output in outputs]).mean()
    epoch_accuracy = torch.stack([output['validation_accuracy'] for output in outputs]).mean()

    return {'validation_loss': epoch_loss.item(), 'validation_accuracy': epoch_accuracy.item()}


def evaluate(model, val_loader):
    metrics = [validation_step(model, batch) for batch in val_loader]
    metrics = validation_epoch_end(metrics)

    return metrics


def get_mnist_dataloaders(batch_size):
    data = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor(), download=True)

    train_data, validation_data = random_split(data, [50000, 10000])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader

def get_cifar_dataloaders(batch_size):
    data = datasets.CIFAR10(root='./datasets', train=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR10(root='./datasets', train=False, transform=transforms.ToTensor())

    train_data, validation_data = random_split(data, [45000, 5000])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader