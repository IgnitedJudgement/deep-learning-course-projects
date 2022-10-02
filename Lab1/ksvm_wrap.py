import numpy as np
from sklearn import svm

from Lab1 import data


class KSVMWrap:
    def __init__(self, X, y, param_svm_c=1, param_svm_gamma='auto', param_svm_kernel='linear'):
        self.classifier = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, kernel=param_svm_kernel)
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def get_scores(self, X):
        return self.classifier.decision_function(X)

    def get_support(self):
        return self.classifier.support_


if __name__ == "__main__":
    np.random.seed(100)

    n_components, n_classes, n_samples = 6, 2, 10

    X, y_true = data.sample_gmm_2d(n_components, n_classes, n_samples)

    model = KSVMWrap(X, y_true, param_svm_kernel='rbf')

    probabilities = model.get_scores(X)
    y_pred = np.where(probabilities > 0, 1, 0)

    accuracy, recall, precision = data.eval_perf_binary(y_true, y_pred)
    AP = data.eval_AP(y_true[y_pred.argsort()])
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, AP: {AP}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: model.predict(x), rect)
    data.graph_data(X, y_true, y_pred, special=model.get_support())
