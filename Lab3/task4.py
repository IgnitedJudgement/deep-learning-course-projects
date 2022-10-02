import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

from Lab3.task1 import Vocabulary
from Lab3.util import get_dataloaders, generate_embedding_matrix, get_frequencies


class ConfigurableModel(nn.Module):
    def __init__(self, embedding, type="lstm", hidden_size=None, num_layers=1, dropout=0, bidirectional=False,
                 loss_function=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam):
        super().__init__()

        self.embedding = embedding
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.types = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }

        if type not in self.types.keys():
            raise ValueError("The given type is currently unsupported!")

        if num_layers < 2 and dropout > 0:
            print("Dropout was excluded since number of layers is equal to 1!")
            dropout = 0

        self.input_size = embedding.weight.shape[1]
        self.hidden_size = int(self.input_size / 2) if hidden_size is None else hidden_size

        self.rnn1 = self.types[type](self.input_size, self.hidden_size,
                                     num_layers, dropout=dropout, bidirectional=bidirectional)
        self.rnn2 = self.types[type](self.hidden_size * (2 if bidirectional else 1), self.hidden_size,
                                     num_layers, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size * (2 if bidirectional else 1), self.hidden_size)
        self.logits = nn.Linear(self.hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.logits.reset_parameters()

    def forward(self, x, lengths=None, hidden=None):
        result = self.embedding(x)
        result = torch.transpose(result, 0, 1)

        result, hidden = self.rnn1(result)
        result, hidden = self.rnn2(result, hidden)

        result = result[-1]
        result = self.fc(result)
        result = torch.relu(result)
        result = self.logits(result)

        return result.squeeze(-1)

    def fit(self, train_dataloader, validation_dataloader, param_delta=1e-4, param_nepochs=8, param_lambda=0,
            param_grad_clip=0.25, param_debug=100, debug=False):
        optimizer = self.optimizer(params=self.parameters(), lr=param_delta, weight_decay=param_lambda)
        self.train()

        writer = SummaryWriter("runs/task4/testing")

        for epoch in range(param_nepochs):
            for index, (features, labels, lengths) in enumerate(train_dataloader):
                outputs = self.forward(features, lengths)

                loss = self.loss_function(outputs, labels.float())
                loss.backward()

                writer.add_scalar("Training loss", loss.item(), epoch * len(train_dataloader) + index)

                nn.utils.clip_grad_norm_(self.parameters(), param_grad_clip)

                optimizer.step()
                optimizer.zero_grad()

            metrics = self.evaluate(validation_dataloader)

            writer.add_scalar("Loss", metrics["loss"], epoch + 1)
            writer.add_scalar("Accuracy", metrics["accuracy"], epoch + 1)
            writer.add_scalar("Precision", metrics["precision"], epoch + 1)
            writer.add_scalar("f1", metrics["f1"], epoch + 1)

            disp = ConfusionMatrixDisplay(metrics["confusion_matrix"])
            disp.plot()

            writer.add_figure("Confusion matrix", disp.figure_, epoch)

            if debug:
                print(f"Epoch {epoch + 1}: validation accuracy = {round(metrics['accuracy'] * 100, 3)}")

        writer.close()

    def evaluate(self, dataloader):
        self.eval()

        labels_true = torch.tensor([])
        labels_pred = torch.tensor([])

        with torch.no_grad():
            for index, (features, labels, lengths) in enumerate(dataloader):
                labels_true = torch.cat((labels_true, labels))
                labels_pred = torch.cat((labels_pred, self.predict(features, lengths)))

        loss = self.loss_function(labels_true, labels_pred)
        accuracy = accuracy_score(labels_true, labels_pred)
        precision = precision_score(labels_true, labels_pred, zero_division=False)
        recall = recall_score(labels_true, labels_pred)
        f1 = f1_score(labels_true, labels_pred)
        confusion = confusion_matrix(labels_true, labels_pred)

        self.train()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion
        }

    def predict(self, x, lengths):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, lengths)).round().int()


if __name__ == "__main__":
    seed = 7052020
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = {
        "train": "./data/train.csv",
        "validation": "./data/validation.csv",
        "test": "./data/test.csv",
        "embedding": "./data/embedding.txt"
    }

    special_symbols = ["<PAD>", "<UNK>"]

    batch_sizes = {
        "train": 10,
        "validate": 32,
        "test": 10
    }

    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(paths, special_symbols, batch_sizes)

    text_frequencies, label_frequencies = get_frequencies("./data/train.csv")

    text_vocabulary = Vocabulary(text_frequencies, special_symbols=["<PAD>", "<UNK>"])
    label_vocabulary = Vocabulary(label_frequencies)

    embedding = generate_embedding_matrix(text_vocabulary, freeze=True, path=paths["embedding"])

    model = ConfigurableModel(embedding, type="rnn", hidden_size=300, num_layers=1, dropout=0, bidirectional=False)
    model.fit(train_dataloader, validation_dataloader, debug=True, param_nepochs=5)

    test_metrics = model.evaluate(test_dataloader)

    print(f"Test accuracy = {round(test_metrics['accuracy'] * 100, 3)}")
