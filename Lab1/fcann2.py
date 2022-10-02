import numpy as np

import data


class ANN:
    def __init__(self, in_features, hidden_features, out_features):
        self.weights = [np.random.randn(in_features, hidden_features), np.random.randn(hidden_features, out_features)]
        self.bias = [np.random.randn(1, hidden_features), np.random.randn(1, out_features)]

    def softmax(self, x):
        max = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - max)
        sum = np.sum(exp, axis=1, keepdims=True)

        return exp / sum

    def relu(self, x):
        return np.maximum(0, x)

    def get_probabilities(self, x, return_intermediate_results=False):
        s1 = np.matmul(x, self.weights[0]) + self.bias[0]
        h1 = self.relu(s1)

        s2 = np.matmul(h1, self.weights[1]) + self.bias[1]
        probabilities = self.softmax(s2)

        if return_intermediate_results:
            return probabilities, s1, h1, s2
        else:
            return probabilities

    def get_loss(self, y_true_onehot, probabilities, param_lambda=0):
        loss = -np.mean(np.log(probabilities[range(len(probabilities)), np.argmax(y_true_onehot, axis=1)]))
        loss += param_lambda * np.sum([np.linalg.norm(weight) for weight in self.weights])

        return loss

    def train(self, X, y_true_onehot, param_niter=10000, param_delta=0.05, param_lambda=1e-3, param_debug=100,
              debug=False):
        N, n = X.shape

        for iter in range(param_niter):
            probabilities, s1, h1, s2 = self.get_probabilities(X, return_intermediate_results=True)
            loss = self.get_loss(y_true_onehot, probabilities, param_lambda)

            if debug and iter % param_debug == 0:
                print(f"Iteration: {iter}, loss: {loss}")

            grad_loss_s2 = np.divide(probabilities - y_true_onehot, N)

            grad_loss_W2 = np.matmul(h1.T, grad_loss_s2) + 2 * param_lambda * self.weights[1]
            grad_loss_b2 = np.sum(grad_loss_s2, axis=0)

            grad_loss_h1 = np.matmul(grad_loss_s2, self.weights[1].T)

            grad_loss_s1 = grad_loss_h1 * (s1 > 0)

            grad_loss_W1 = np.matmul(X.T, grad_loss_s1) + 2 * param_lambda * self.weights[0]
            grad_loss_b1 = np.sum(grad_loss_s1, axis=0)

            self.weights[0] -= param_delta * grad_loss_W1
            self.weights[1] -= param_delta * grad_loss_W2

            self.bias[0] -= param_delta * grad_loss_b1
            self.bias[1] -= param_delta * grad_loss_b2

    def evaluate(self, x):
        return self.get_probabilities(x)

    def classify(self, probabilities):
        return np.argmax(probabilities, axis=1)

    def predict(self, x):
        return np.argmax(self.get_probabilities(x), axis=1)


if __name__ == "__main__":
    np.random.seed(100)

    n_components, n_classes, n_samples = 6, 2, 10
    in_features, hidden_features, out_features = 2, 6, n_classes

    X, y_true = data.sample_gmm_2d(n_components, n_classes, n_samples)
    y_true_onehot = data.class_to_onehot(y_true)

    model = ANN(in_features, hidden_features, out_features)
    model.train(X, y_true_onehot, debug=True)

    probabilities = model.evaluate(X)
    y_pred = model.classify(probabilities)

    accuracy, recall, precision = data.eval_perf_binary(y_true, y_pred)
    AP = data.eval_AP(y_true[np.amax(probabilities, axis=1).argsort()])
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, AP: {AP}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: model.predict(x), rect)
    data.graph_data(X, y_true, y_pred)
