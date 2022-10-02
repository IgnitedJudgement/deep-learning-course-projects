import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from util import get_cifar_dataloaders

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, eval_mode=False):
        if eval_mode:
            self.eval()

        result = self.conv1(x)
        result = torch.relu(result)
        result = self.pool1(result)

        result = self.conv2(result)
        result = torch.relu(result)
        result = self.pool2(result)

        result = result.view((result.shape[0], -1))

        result = self.fc1(result)
        result = torch.relu(result)
        result = self.fc2(result)
        result = torch.relu(result)
        result = self.fc3(result)

        if eval_mode:
            self.train()

        return result


def train(model, train_dataloader, validation_dataloader, param_delta=0.1, param_nepochs=8, param_lambda=1e-3):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)

    writer = SummaryWriter()

    for epoch in range(param_nepochs):
        batch_losses = []
        y_true_train, y_pred_train = [], []

        for index, (features, labels) in enumerate(train_dataloader):
            outputs = model.forward(features)

            loss = model.loss_function(outputs, labels)
            batch_losses.append(loss.item())
            loss.backward()

            y_true_train.extend(labels)
            y_pred_train.extend(classify(outputs))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch * len(train_dataloader) + index)
            writer.add_scalar("Loss", loss, epoch * len(train_dataloader) + index)

        validation_features, validation_labels = next(iter(validation_dataloader))
        validation_outputs = model.forward(validation_features, eval_mode=True)

        train_cm, train_accuracy, train_recall, train_precision = evaluate(model, y_true_train, y_pred_train)
        cm, accuracy, recall, precision = evaluate(model, validation_labels, classify(validation_outputs))

        writer.add_scalars("Accuracy", {
            'train': train_accuracy,
            'validation': accuracy
        }, epoch)

        writer.add_scalars("Recall", {
            'train': train_precision,
            'validation': precision
        }, epoch)

        writer.add_scalars("Precision", {
            'train': train_recall,
            'validation': recall
        }, epoch)

        writer.add_scalar("Average loss", np.mean(batch_losses), epoch)
        writer.add_image("Filters", torchvision.utils.make_grid(model.conv1.weight), epoch)

    writer.close()


def classify(outputs):
    return torch.argmax(outputs, axis=1)


def evaluate(model, y_true, y_pred):
    model.eval()
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    model.train()

    return cm, accuracy, recall, precision


if __name__ == "__main__":
    batch_size, n_epochs = 50, 8

    train_dataloader, validation_dataloader, test_dataloader = get_cifar_dataloaders(batch_size=batch_size)

    net = ConvolutionalModel()
    train(net, train_dataloader, validation_dataloader, param_nepochs=n_epochs)
