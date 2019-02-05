from keras.datasets import mnist, cifar10
import numpy as np
from keras.utils import to_categorical

def inbalanced_mnist(inbalance_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    filters = np.zeros(X_train.shape[0], dtype=np.bool)
    for i in range(10):
        current_filters = y_train == i
        if i % 2 == 0:
            current_filters = np.logical_and(current_filters, np.cumsum(current_filters)<=inbalance_size*6)
        filters = np.logical_or(filters, current_filters)
    X_train, y_train = X_train[filters], y_train[filters]

    filters = np.zeros(X_test.shape[0], dtype=np.bool)
    for i in range(10):
        current_filters = y_test == i
        if i % 2 == 0:
            current_filters = np.logical_and(current_filters, np.cumsum(current_filters)<=inbalance_size)
        filters = np.logical_or(filters, current_filters)
    X_test, y_test = X_test[filters], y_test[filters]

    X_train, X_test = np.expand_dims(X_train / 255.0, axis=-1), np.expand_dims(X_test / 255.0, axis=-1)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # dummy for regularization term
    dummy_train, dummy_test = np.zeros((X_train.shape[0], 1), dtype=np.float32), np.zeros((X_test.shape[0], 1), dtype=np.float32)
    y_train = np.concatenate([y_train, dummy_train], axis=-1)
    y_test = np.concatenate([y_test, dummy_test], axis=-1)

    return (X_train, y_train), (X_test, y_test)

def inbalanced_cifar(inbalance_size):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    filters = np.zeros(X_train.shape[0], dtype=np.bool)
    for i in range(10):
        current_filters = y_train[:,0] == i
        if i % 2 == 0:
            current_filters = np.logical_and(current_filters, np.cumsum(current_filters)<=inbalance_size*5)
        filters = np.logical_or(filters, current_filters)
    X_train, y_train = X_train[filters], y_train[filters]

    filters = np.zeros(X_test.shape[0], dtype=np.bool)
    for i in range(10):
        current_filters = y_test[:,0] == i
        if i % 2 == 0:
            current_filters = np.logical_and(current_filters, np.cumsum(current_filters)<=inbalance_size)
        filters = np.logical_or(filters, current_filters)
    X_test, y_test = X_test[filters], y_test[filters]

    # ensure batch size is divisible by 8
    n_train = X_train.shape[0] // 8 * 8
    n_test = X_test.shape[0] // 8 * 8
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # dummy for regularization term
    dummy_train, dummy_test = np.zeros((X_train.shape[0], 1), dtype=np.float32), np.zeros((X_test.shape[0], 1), dtype=np.float32)
    y_train = np.concatenate([y_train, dummy_train], axis=-1)
    y_test = np.concatenate([y_test, dummy_test], axis=-1)

    return (X_train, y_train), (X_test, y_test)
