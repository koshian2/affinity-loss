from keras import layers
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, Callback
import keras.backend as K
from affnity_loss import *
from datasets import inbalanced_mnist

import numpy as np
import os, tarfile
import matplotlib.pyplot as plt


def conv_bn_relu(input, ch):
    x = layers.Conv2D(ch, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def create_models():
    input = layers.Input((28,28,1))
    x = conv_bn_relu(input, 32)
    x = layers.AveragePooling2D(2)(x)
    x = conv_bn_relu(x, 64)
    x = layers.AveragePooling2D(2)(x)
    x = conv_bn_relu(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, name="latent_features", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = ClusteringAffinity(10, 1, 5.0)(x)

    return Model(input, x)

def acc(y_true_plusone, y_pred_plusone):
    y_true = K.argmax(y_true_plusone[:, :-1], axis=-1)
    y_pred = K.argmax(y_pred_plusone[:, :-1], axis=-1)
    equals = K.cast(K.equal(y_true, y_pred), "float")
    return K.mean(equals)

def step_decay(epoch):
    x = 1e-3
    if epoch >= 75: x /= 5.0
    return x

class EmbeddingCallback(Callback):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train, self.y_train = X_train, y_train[:,:10]
        self.X_test, self.y_test = X_test, y_test[:,:10]

    def plot(self, X, y, title):
        latent_model = Model(self.model.input, self.model.get_layer("latent_features").output)
        embedding = latent_model.predict(X, batch_size=128)
        plt.clf()
        cmap = plt.get_cmap("tab10")
        for i in range(y.shape[1]):
            filtered = y[:, i] == 1.0
            plt.scatter(embedding[filtered, 0], embedding[filtered, 1], marker="$"+str(i)+"$", alpha=0.5, color=cmap(i), )
        plt.savefig(title+".png")

    def on_epoch_end(self, epoch, logs):
        output_dir = "mnist_inbalanced"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.plot(self.X_train, self.y_train, f"{output_dir}/mnist_train_{epoch:03}")
        self.plot(self.X_test, self.y_test, f"{output_dir}/mnist_test_{epoch:03}")

def train(inbalance_size):
    (X_train, y_train), (X_test, y_test) = inbalanced_mnist(inbalance_size)

    model = create_models()
    model.compile("adam", affinity_loss(0.75), [acc])

    scheduler = LearningRateScheduler(step_decay)
    cb = EmbeddingCallback(model, X_train, X_test, y_train, y_test)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[cb, scheduler],
                        batch_size=128, epochs=200, verbose=1).history

    with tarfile.open("mnist_inbalanced.tar", "w") as tar:
        tar.add("mnist_inbalanced")

if __name__ == "__main__":
    train(100)
