import numpy as np

import data


class BinaryLogisticRegression:
    def __init__(self, in_features):
        self.weight = np.random.randn(in_features)
        self.bias = 0

    def sigmoid(self, x):
        return np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))

    def get_scores(self, x):
        return np.matmul(x, self.weight.T) + self.bias

    def get_probabilities(self, x):
        return self.sigmoid(self.get_scores(x))

    def get_loss(self, y_true, probabilities, param_lambda=0):
        loss = -np.mean(np.where(y_true == 1, np.log(probabilities), np.log(1 - probabilities)))
        loss += param_lambda * np.linalg.norm(self.weight)

        return loss

    def train(self, X, y_true, param_niter=10000, param_delta=0.05, param_lambda=1e-3, param_debug=10, debug=False):
        N, n = X.shape

        for iter in range(param_niter):
            scores = self.get_scores(X)
            probabilities = self.sigmoid(scores)

            loss = self.get_loss(y_true, probabilities, param_lambda)

            if debug and iter % param_debug == 0:
                print(f"Iteration: {iter}, loss: {loss}")

            grad_loss_scores = probabilities - y_true

            grad_loss_weight = np.divide(np.matmul(grad_loss_scores.T, X), N)
            grad_loss_bias = np.mean(grad_loss_scores)

            self.weight -= param_delta * grad_loss_weight
            self.bias -= param_delta * grad_loss_bias

    def predict(self, x):
        return np.where(self.get_probabilities(x) >= 0.5, 1, 0)


if __name__ == "__main__":
    np.random.seed(100)
    in_features, n_classes, n_samples = 2, 2, 100

    X, y_true = data.sample_gauss_2d(n_classes, n_samples)

    model = BinaryLogisticRegression(in_features)
    model.train(X, y_true)

    probabilities = model.get_probabilities(X)
    y_pred = model.predict(X)

    accuracy, recall, precision = data.eval_perf_binary(y_true, y_pred)
    AP = data.eval_AP(y_true[probabilities.argsort()])
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, AP: {AP}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: model.predict(x), rect)
    data.graph_data(X, y_true, y_pred)
