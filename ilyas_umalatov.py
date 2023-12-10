import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
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

#Tarea 1A
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
    def __init__(self, T=40, A=20):
        self.T = T
        self.A = A
        self.clfs = []
        self.train_time = None

    def fit(self, X, Y, verbose=False):
        start_time = time.time()
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
        end_time = time.time()
        self.train_time = end_time - start_time

    def predict(self, X):
        clf_preds = [alpha * clf.predict(X) for clf, alpha in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

#Tarea 1B
def adaboost_test(class_label, T, A):
    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convert labels to -1 and 1
    Y_train = np.where(Y_train == class_label, 1, -1)
    Y_test = np.where(Y_test == class_label, 1, -1)

    # Create and train Adaboost classifier
    clf = Adaboost(T, A)

    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)

    # Print accuracy scores
    print(f'Train accuracy: {train_accuracy * 100}%')
    print(f'Test accuracy: {test_accuracy * 100}%')
    print(f'Time: {clf.train_time} s')

    # Print additional information
    for i, (clf, alpha) in enumerate(clf.clfs, 1):
        print(f'Añadido clasificador {i}: {clf.feature_index}, {clf.threshold}, {clf.polarity}, {alpha}')

#Tarea 1C
def adaboost_grafic():
    # Define the values of T and A to test
    T_values = [10, 20, 30, 40, 50]
    A_values = [10, 20, 30, 40, 50]

    # Initialize empty lists for storing results
    train_accuracies = []
    test_accuracies = []
    train_times = []

    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convert labels to -1 and 1
    Y_train = np.where(Y_train == 0, 1, -1)
    Y_test = np.where(Y_test == 0, 1, -1)

    for T in T_values:
        for A in A_values:
            # Create and train Adaboost classifier
            clf = Adaboost(T, A)
            clf.fit(X_train, Y_train)

            # Make predictions
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            # Calculate accuracy scores
            train_accuracy = accuracy_score(Y_train, y_train_pred)
            test_accuracy = accuracy_score(Y_test, y_test_pred)

            # Store results
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            train_times.append(clf.train_time)

    # Reshape the results
    train_accuracies = np.array(train_accuracies).reshape(len(T_values), len(A_values))
    train_times = np.array(train_times).reshape(len(T_values), len(A_values))

    # Create a figure and a subplot
    fig, ax1 = plt.subplots()

    # Plot accuracy
    for i, A in enumerate(A_values):
        ax1.plot(T_values, train_accuracies[:, i], label=f'Accuracy A={A}')
    ax1.set_xlabel('T (number of iterations)')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the same plot
    ax2 = ax1.twinx()

    # Plot training time
    for i, A in enumerate(A_values):
        ax2.bar(T_values, train_times[:, i], label=f'Training time A={A}', alpha=0.5)
    ax2.set_ylabel('Training time (s)')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    ax1.set_xticks(T_values)
    ax2.set_xticks(T_values)
    plt.show()

#Tarea 1D
class Adaboost2:
    def __init__(self, T=40, A=20):
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
    def __init__(self, num_classes):
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
    clf = MultiClassClassifier(10)

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
            print(f'\tAñadido clasificador {j}: {weak_clf.feature_index}, {weak_clf.threshold}, {weak_clf.polarity}, {alpha}')

#Tarea 2A
def adaboost3_test():
    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Create and train AdaBoost classifier
    clf = AdaBoostClassifier()

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

#Tarea 2D
def keras_mlp_test():
    # Load MNIST data
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convert labels to categorical
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # Create MLP model
    model = Sequential()
    model.add(Flatten(input_shape=(28*28,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the model
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

    # Print accuracy scores
    print(f'Train accuracy: {train_acc * 100}%')
    print(f'Test accuracy: {test_acc * 100}%')

if __name__ == '__main__':
    #Tarea 1C
    adaboost_grafic()
