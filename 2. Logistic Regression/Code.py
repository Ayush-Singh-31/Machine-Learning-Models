import numpy as np

class Perceptron:
    def __init__(self, learningRate=0.01, iterations=1000, fitIntercept=True):
        self.lr = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept

    def _add_intercept(self, X):
        if self.fitIntercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack((intercept, X))
        return X

    def fit(self, X, y):
        X = self._add_intercept(X)
        self.w = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                activation = 1 if np.dot(self.w, xi) >= 0 else 0
                self.w += self.lr * (yi - activation) * xi

    def predict(self, X):
        X = self._add_intercept(X)
        raw = X.dot(self.w)
        return np.where(raw >= 0, 1, 0)


class BinaryLogisticRegression:
    def __init__(self, learningRate=0.1, iterations=1000, fitIntercept=True):
        self.lr = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept

    def _add_intercept(self, X):
        if self.fitIntercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack((intercept, X))
        return X

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = self._add_intercept(X)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.iterations):
            z = X.dot(self.w)
            h = self._sigmoid(z)
            gradient = X.T.dot(y - h)  
            self.w += self.lr * gradient / n_samples

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.w))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    def __init__(self, learningRate=0.1, iterations=1000, fitIntercept=True):
        self.lr = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept

    def _add_intercept(self, X):
        if self.fitIntercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack((intercept, X))
        return X

    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = self._add_intercept(X)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_indices = np.searchsorted(self.classes_, y)
        self.W = np.zeros((n_features, n_classes))
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_indices] = 1
        for _ in range(self.iterations):
            scores = X.dot(self.W)               
            P = self._softmax(scores)           
            gradient = X.T.dot(Y_onehot - P)    
            self.W += self.lr * gradient / n_samples

    def predict_proba(self, X):
        X = self._add_intercept(X)
        scores = X.dot(self.W)
        return self._softmax(scores)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]