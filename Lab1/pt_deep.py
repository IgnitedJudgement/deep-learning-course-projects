import numpy as np
import torch, torch.nn as nn

from Lab1 import data


class Deep(nn.Module):
    def __init__(self, arhitecture, activation_function=None):
        super().__init__()

        self.arhitecture = arhitecture

        self.activation_function = activation_function if activation_function else lambda x: x

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(arhitecture[i], arhitecture[i + 1], dtype=torch.float32),
                          requires_grad=True) for i in
             range(len(arhitecture) - 1)])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.randn(1, arhitecture[i + 1], dtype=torch.float32),
                          requires_grad=True) for i in
             range(len(arhitecture) - 1)])

    def forward(self, X):
        for weight, bias in zip(self.weights, self.biases):
            X = self.activation_function(X.mm(weight) + bias)

        probabilities = X.softmax(dim=1)

        return probabilities

    def get_loss(self, X, y):
        probabilities = self.forward(X)
        log = torch.log(probabilities + 1e-13) * y
        sum = torch.sum(log, dim=1)
        mean = torch.mean(sum)

        return -mean


def train(model, X, y, param_niter=10000, param_delta=0.1, param_debug=100, param_lambda=1e-4, debug=False):
    optimiser = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    for iter in range(param_niter):
        loss = model.get_loss(X, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if iter % param_debug == 0 and debug:
            print(f"iteration: {iter}, loss:: {loss}")


def evaluate(model, X):
    return model.forward(X).detach().cpu().numpy()


def predict(model, X):
    return np.argmax(evaluate(model, X), axis=1)


def classify(probabilities):
    return np.argmax(probabilities, axis=1)


def count_params(model):
    number_of_layers, counter, bias = len(model.arhitecture), 1, False
    n_param = (model.arhitecture[0] + 1) * model.arhitecture[1]

    print("Dimensions of layers:")

    for parameter in model.parameters():
        x, y = parameter.data.shape
        print(f"{'W' if not bias else 'b'}{counter} dimensions: {x}x{y}")

        if counter == number_of_layers - 1:
            counter = 1
            bias = True
        else:
            n_param += (x + 1) * y
            counter += 1

    print(f"Total number of parameters: {n_param}")


if __name__ == "__main__":
    np.random.seed(100)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    n_components, n_classes, n_samples = 6, 2, 10

    X, y = data.sample_gmm_2d(n_components, n_classes, n_samples)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.class_to_onehot(y), dtype=torch.float32, device=device)

    model = Deep([2, 10, 10, n_classes], activation_function=torch.sigmoid).to(device)
    train(model, X_tensor, y_tensor, param_niter=5000, param_delta=0.15, debug=False)
    probabilities = evaluate(model, X_tensor)
    y_pred = classify(probabilities)

    accuracy, recall, precision = data.eval_perf_binary(y, y_pred)
    AP = data.eval_AP(y[np.amax(probabilities, axis=1).argsort()])
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, AP: {AP}")

    count_params(model)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: predict(model, torch.tensor(x, dtype=torch.float32, device=device)), rect)
    data.graph_data(X, y, y_pred)
