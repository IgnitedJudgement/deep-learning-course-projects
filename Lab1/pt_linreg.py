import numpy as np
import torch
import torch.nn as nn

from Lab1 import data


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.a = nn.parameter.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.b = nn.parameter.Parameter(torch.randn(1, out_features), requires_grad=True)

    def forward(self, X):
        return X.mm(self.a) + self.b

    def get_loss(self, X, y):
        return torch.mean(torch.square(torch.subtract(self.forward(X), y)))


def train(model, X, y, param_niter=1000, param_delta=0.05, param_debug=100, debug=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    for iter in range(param_niter):
        loss = model.get_loss(X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % param_debug == 0 and debug:
            print(f"iteration: {iter}, loss:: {loss}, a: {model.a}, b: {model.b}")


def evaluate(model, X):
    return model.forward(X).detach().cpu().numpy()


def predict(model, X):
    return np.argmax(evaluate(model, X), axis=1)


def classify(probabilities):
    return np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    np.random.seed(100)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    X, y = data.sample_gauss_2d(2, 100)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.class_to_onehot(y), dtype=torch.float32, device=device)

    n_samples, n_features = X.shape

    model = LinearRegression(n_features, n_features).to(device)
    train(model, X_tensor, y_tensor, param_niter=100)
    probabilities = evaluate(model, X_tensor)
    y_pred = classify(probabilities)

    accuracy, recall, precision = data.eval_perf_binary(y, y_pred)
    AP = data.eval_AP(y[np.amax(probabilities, axis=1).argsort()])
    print(accuracy, recall, precision, AP)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: predict(model, torch.tensor(x, dtype=torch.float32, device=device)), rect)
    data.graph_data(X, y, y_pred)
