import numpy as np

import data


class LogisticRegression:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.zeros(out_features)

    def softmax(self, x):
        max = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - max)
        sum = np.sum(exp, axis=1, keepdims=True)

        return exp / sum

    def get_scores(self, x):
        return np.matmul(x, self.weights) + self.bias

    def get_probabilities(self, x):
        return self.softmax(self.get_scores(x))

    def get_loss(self, y_true_onehot, probabilities, param_lambda=0):
        loss = -np.mean(np.log(probabilities[range(len(probabilities)), np.argmax(y_true_onehot, axis=1)]))
        loss += np.linalg.norm(self.weights)

        return loss

    def train(self, X, y_true_onehot, param_niter=10000, param_delta=0.05, param_lambda=1e-3, param_debug=10,
              debug=False):
        N, n = X.shape

        for iter in range(param_niter):
            scores = self.get_scores(X)
            probabilities = self.softmax(scores)

            loss = self.get_loss(y_true_onehot, probabilities, param_lambda)

            if debug and iter % param_debug == 0:
                print(f"Iteration: {iter}, loss: {loss}")

            grad_loss_scores = probabilities - y_true_onehot

            grad_loss_weights = np.divide(np.matmul(grad_loss_scores.T, X), N).T
            grad_loss_bias = np.divide(np.sum(grad_loss_scores.T, axis=1), N)

            self.weights -= param_delta * grad_loss_weights
            self.bias -= param_delta * grad_loss_bias

    def predict(self, X):
        return np.argmax(self.get_probabilities(X), axis=1)


if __name__ == "__main__":
    np.random.seed(100)
    in_features, n_classes, n_samples = 2, 3, 100

    X, y_true = data.sample_gauss_2d(n_classes, n_samples)
    y_true_onehot = data.class_to_onehot(y_true)

    model = LogisticRegression(in_features, n_classes)
    model.train(X, y_true_onehot)

    probabilities = model.get_probabilities(X)
    y_pred = model.predict(X)

    accuracy, recall, precision = data.eval_perf_binary(y_true, y_pred)
    AP = data.eval_AP(y_true[np.amax(probabilities, axis=1).argsort()])
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, AP: {AP}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: model.predict(x), rect)
    data.graph_data(X, y_true, y_pred)
