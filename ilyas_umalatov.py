import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
from sklearn.metrics import accuracy_score
import time


def load_MNIST_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")

    return X_train, Y_train, X_test, Y_test

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1 if np.random.uniform() < 0.5 else -1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] > self.threshold] = -1
        return predictions

class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.clfs = []

    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape
        W = np.full(n_samples, (1 / n_samples))
        for _ in range(self.T):
            best_clf, best_error, best_predictions = None, float('inf'), None
            for _ in range(self.A):
                clf = DecisionStump(n_features)
                predictions = clf.predict(X)
                error = W[Y != predictions].sum()
                if error < best_error:
                    best_clf, best_error, best_predictions = clf, error, predictions
            alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
            W *= np.exp(-alpha * Y * best_predictions)
            W /= np.sum(W)
            self.clfs.append((best_clf, alpha))

    def predict(self, X):
        clf_preds = [alpha * clf.predict(X) for clf, alpha in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

def adaboost_test(class_label, T, A):
    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convert labels to -1 and 1
    Y_train = np.where(Y_train == class_label, 1, -1)
    Y_test = np.where(Y_test == class_label, 1, -1)

    # Create and train Adaboost classifier
    clf = Adaboost(T, A)

    start_time = time.time()
    clf.fit(X_train, Y_train)
    end_time = time.time()

    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    # Print accuracy scores
    print(f'Train accuracy: {train_accuracy * 100}%')
    print(f'Test accuracy: {test_accuracy * 100}%')
    print(f'Time: {end_time - start_time} s')

    # Print additional information
    for i, (clf, alpha) in enumerate(clf.clfs, 1):
        print(f'Añadido clasificador {i}: {clf.dim}, {clf.thresh}, {clf.polarity}, {alpha}')

class Adaboost2:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.clfs = []

    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape
        W = np.full(n_samples, (1 / n_samples))
        for _ in range(self.T):
            best_clf, best_error, best_predictions = None, float('inf'), None
            for _ in range(self.A):
                clf = DecisionStump(n_features)
                predictions = clf.predict(X)
                error = W[Y != predictions].sum()
                if error < best_error:
                    best_clf, best_error, best_predictions = clf, error, predictions
            alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
            W *= np.exp(-alpha * Y * best_predictions)
            W /= np.sum(W)
            self.clfs.append((best_clf, alpha))

    def predict(self, X):
        # Calculate the weighted sum of the weak classifiers' predictions
        weighted_sum = sum(alpha * clf.predict(X) for clf, alpha in self.clfs)
        # Return the weighted sum without taking the sign
        return weighted_sum

class MultiClassClassifier:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.classifiers = [Adaboost2() for _ in range(num_classes)]

    def fit(self, X, Y):
        for i in range(self.num_classes):
            Y_binary = np.where(Y == i, 1, -1)
            self.classifiers[i].fit(X, Y_binary)

    def predict(self, X):
        scores = np.array([clf.predict(X) for clf in self.classifiers])
        predictions = np.argmax(scores, axis=0)
        return predictions

def adaboost_test2(T, A):
    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Create and train Adaboost classifier
    clf = MultiClassClassifier(T, A)

    start_time = time.time()
    clf.fit(X_train, Y_train)
    end_time = time.time()

    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    # Print accuracy scores
    print(f'Train accuracy: {train_accuracy * 100}%')
    print(f'Test accuracy: {test_accuracy * 100}%')
    print(f'Time: {end_time - start_time} s')

    # Print additional information
    for i, clf in enumerate(clf.classifiers, 1):
        print(f'Clasificador {i}:')
        for j, (weak_clf, alpha) in enumerate(clf.clfs, 1):
            print(f'\tAñadido clasificador {j}: {weak_clf.dim}, {weak_clf.thresh}, {weak_clf.polarity}, {alpha}')
