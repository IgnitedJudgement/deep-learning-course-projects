from pathlib import Path

import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from util import evaluate, draw_conv_filters, get_mnist_dataloaders


class ConvolutionalModel(nn.Module):
    # 1, 16, 2, 32, 2, 512, 10
    def __init__(self, in_channels, conv1_width, pool1_width, conv2_width, pool2_width, fc3_width, class_count,
                 loss_function=nn.CrossEntropyLoss()):
        super().__init__()

        self.loss_function = loss_function

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_width, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_width, stride=pool1_width)

        pool1_out = conv1_width

        self.conv2 = nn.Conv2d(in_channels=pool1_out, out_channels=conv2_width, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_width, stride=pool2_width)

        fc3_in = conv2_width * 7 * 7

        self.fc3 = nn.Linear(in_features=fc3_in, out_features=fc3_width)
        self.logits = nn.Linear(in_features=fc3_width, out_features=class_count)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.logits.reset_parameters()

    def forward(self, x):
        result = self.conv1(x)
        result = self.pool1(result)
        result = torch.relu(result)

        result = self.conv2(result)
        result = self.pool2(result)
        result = torch.relu(result)

        result = result.view((result.shape[0], -1))

        result = self.fc3(result)
        result = torch.relu(result)
        result = self.logits(result)

        return result

    def train(self, train_dataloader, validation_dataloader, param_delta=0.1, param_nepochs=8,
              param_lambda=1e-3, save_dir=None, param_debug=100, debug=False):
        optimizer = torch.optim.SGD(params=self.parameters(), lr=param_delta, weight_decay=param_lambda)

        writer = SummaryWriter()

        metrics_list = []

        for epoch in range(param_nepochs):
            batch_losses = []

            for index, (features, labels) in enumerate(train_dataloader):
                outputs = self.forward(features)

                loss = self.loss_function(outputs, labels)
                loss.backward()
                batch_losses.append(loss.item())

                writer.add_scalar("Training loss", loss.item(), epoch * len(train_dataloader) + index)

                optimizer.step()
                optimizer.zero_grad()

                if index % 100 == 0:
                    writer.add_image("Filters", torchvision.utils.make_grid(self.conv1.weight), 0)

            metrics = evaluate(self, validation_dataloader)
            metrics_list.append(metrics)

            writer.add_scalar("Average training loss", np.mean(batch_losses), epoch)
            writer.add_scalar("Validation loss", round(metrics['validation_loss'], 3), epoch)
            writer.add_scalar("Validation accuracy", round(metrics['validation_accuracy'] * 100, 3), epoch)

            if debug:
                print(f"Epoch {epoch + 1}, validation loss: {round(metrics['validation_loss'], 3)}"
                      f", validation accuracy: {round(metrics['validation_accuracy'] * 100, 3)}%")

        writer.close()

        return metrics_list


if __name__ == "__main__":
    SAVE_DIR = Path(__file__).parent / 'out_pt_conv/reg_0'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyper parameters
    batch_size, n_epochs = 50, 8
    in_channels = 1
    conv1_width = 16
    pool1_width = 2
    conv2_width = 32
    pool2_width = 2
    fc3_width = 512
    class_count = 10

    train_dataloader, validation_dataloader, test_dataloader = get_mnist_dataloaders(batch_size=batch_size)

    net = ConvolutionalModel(in_channels, conv1_width, pool1_width, conv2_width, pool2_width, fc3_width, class_count)
    metrics_list = net.train(train_dataloader, validation_dataloader, param_nepochs=n_epochs,
                             debug=True, save_dir=SAVE_DIR)

    plt.plot([metrics['validation_loss'] for metrics in metrics_list])
    plt.xlabel("Epohs")
    plt.ylabel("Average loss")
    plt.show()
