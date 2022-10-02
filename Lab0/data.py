import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    min_x, max_x, min_y, max_y, cov_scale = 0, 10, 0, 10, 5
    mean, sigma = None, None

    def __init__(self):
        delta_x, delta_y = self.max_x - self.min_x, self.max_y - self.min_y

        mean = np.random.random_sample(2) * (delta_x, delta_y)
        mean += (self.min_x, self.min_y)
        self.mean = mean

        eigenvalues = pow((np.random.random_sample(2)) * (delta_x / self.cov_scale, delta_y / self.cov_scale), 2)

        theta = np.random.random_sample() * 2 * np.pi

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        sigma = np.matmul(np.matmul(rotation_matrix.T, np.diag(eigenvalues)), rotation_matrix)
        self.sigma = sigma

    def get_sample(self, N):
        return np.random.multivariate_normal(self.mean, self.sigma, N)


def sample_gauss_2d(C, N):
    gauss_list = np.array([Random2DGaussian() for _ in range(C)])
    y_true_list = np.array([i for i in range(C)])

    X = np.vstack([gauss.get_sample(N) for gauss in gauss_list])
    y_true = np.hstack([[y_true] * N for y_true in y_true_list])

    return X, y_true


def sample_gmm_2d(K, C, N):
    # gauss_list = np.array([Random2DGaussian() for _ in range(K)])
    # y_true_list = np.array([np.random.randint(C) for _ in range(K)])

    gauss_list = []
    y_true_list = []

    for i in range(K):
        gauss_list.append(Random2DGaussian())
        y_true_list.append(np.random.randint(C))

    X = np.vstack([gauss.get_sample(N) for gauss in gauss_list])
    y_true = np.hstack([[y_true] * N for y_true in y_true_list])

    return X, y_true


def class_to_onehot(y_true):
    y_onehot = np.zeros((len(y_true), max(y_true) + 1))
    y_onehot[range(len(y_true)), y_true] = 1
    return y_onehot


def eval_perf_binary(y_true, y_pred):
    tp = sum(np.logical_and(y_pred == y_true, y_true == True))
    fn = sum(np.logical_and(y_pred != y_true, y_true == True))
    tn = sum(np.logical_and(y_pred == y_true, y_true == False))
    fp = sum(np.logical_and(y_pred != y_true, y_true == False))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)

    return accuracy, recall, precision


def eval_AP(y_true_ranked):
    N = len(y_true_ranked)
    tp = sum(y_true_ranked)
    fp = N - tp

    total_precision = 0

    for y_true in y_true_ranked:
        precision = 0 if tp + fp == 0 else tp / (tp + fp)

        if y_true:
            total_precision += precision

        tp -= y_true
        fp -= not y_true

    return total_precision / sum(y_true_ranked)


def graph_data(X, y_true, y_pred, special=None):
    palette = ([0.7, 0.7, 0.7], [0.2, 0.2, 0.2], [1, 1, 1])
    colors = np.tile([0.0, 0.0, 0.0], (y_true.shape[0], 1))

    for i in range(len(palette)):
        colors[y_true == i] = palette[i]

    correct = (y_true == y_pred)
    incorrect = (y_true != y_pred)

    plt.scatter(X[correct, 0], X[correct, 1], c=colors[correct], marker='o')
    plt.scatter(X[incorrect, 0], X[incorrect, 1], c=colors[incorrect], marker='s')

    plt.show()


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


if __name__ == "__main__":
    # np.random.seed(100)
    # G = Random2DGaussian()
    # X = G.get_sample(100)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    ###

    # np.random.seed(100)
    #
    # # get the training dataset
    # X, y_true = sample_gauss_2d(2, 100)
    #
    # # get the class predictions
    # y_pred = myDummyDecision(X) > 0.5
    #
    # # graph the data points
    # graph_data(X, y_true, y_pred)
    #
    # # show the results
    # plt.show()

    ###

    # np.random.seed(100)
    #
    # # get the training dataset
    # X, y_true = sample_gauss_2d(2, 100)
    #
    # # get the class predictions
    # y_pred = myDummyDecision(X) > 0.5
    #
    # # graph the decision surface
    # rect = (np.min(X, axis=0), np.max(X, axis=0))
    # graph_surface(myDummyDecision, rect, offset=0)
    #
    # # graph the data points
    # graph_data(X, y_true, y_pred)
    #
    # # show the results
    # plt.show()

    ###

    np.random.seed(100)

    # get the training dataset
    X, y_true = sample_gmm_2d(6, 2, 10)

    # get the class predictions
    y_pred = myDummyDecision(X) > 0.5

    # graph the decision surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, offset=0)

    # graph the data points
    graph_data(X, y_true, y_pred)

    # show the results
    plt.show()
