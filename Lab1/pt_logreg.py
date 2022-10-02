import numpy as np
import torch, torch.nn as nn

from Lab1 import data


class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weights = nn.parameter.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.randn(out_features), requires_grad=True)
    
    def forward(self, X):
        outputs = X.mm(self.weights) + self.bias
        probabilities = outputs.softmax(dim=1)

        return probabilities

    def get_loss(self, X, y):
        probabilities = self.forward(X)
        log = y * torch.log(probabilities + 1e-13)
        sum = torch.sum(log, dim=1)
        mean = torch.mean(sum)

        return -mean


def train(model, X, y, param_niter=1000, param_delta=0.5, param_lambda=0.001, param_debug=100, debug=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    for iter in range(param_niter):
        loss = model.get_loss(X, y) + param_lambda * torch.linalg.matrix_norm(model.weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % param_debug == 0 and debug:
            print(f"iteration: {iter}, loss:: {loss}")


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

    n_components, n_classes, n_samples = 2, 3, 100

    # X, y = data.sample_gmm_2d(n_components, n_classes, n_samples)
    X, y = data.sample_gauss_2d(n_classes, n_samples)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(data.class_to_onehot(y), dtype=torch.float32, device=device)

    model = LogisticRegression(X.shape[1], n_classes).to(device)
    train(model, X_tensor, y_tensor)
    probabilities = evaluate(model, X_tensor)
    y_pred = classify(probabilities)

    accuracy, recall, precision = data.eval_perf_binary(y, y_pred)
    AP = data.eval_AP(y[np.amax(probabilities, axis=1).argsort()])
    print(accuracy, recall, precision, AP)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: predict(model, torch.tensor(x, dtype=torch.float32, device=device)), rect)
    data.graph_data(X, y, y_pred)
