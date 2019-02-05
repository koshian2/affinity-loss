import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, Callback
import keras.backend as K
from affnity_loss import *
from datasets import inbalanced_mnist

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import f1_score


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
    x = layers.BatchNormalization()(x)
    x = ClusteringAffinity(10, 1, 10.0)(x)

    return Model(input, x)

def acc(y_true_plusone, y_pred_plusone):
    y_true = K.argmax(y_true_plusone[:, :-1], axis=-1)
    y_pred = K.argmax(y_pred_plusone[:, :-1], axis=-1)
    equals = K.cast(K.equal(y_true, y_pred), "float")
    return K.mean(equals)

def step_decay(epoch):
    x = 1e-3
    if epoch >= 50: x /= 5.0
    if epoch >= 80: x /= 5.0
    return x

class F1Callback(Callback):
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test_label = np.argmax(y_test, axis=-1)
        self.f1_log = []

    def on_epoch_end(self, epoch, logs):
        if epoch % 25 == 0: print(epoch, "ends")
        y_pred = self.model.predict(self.X_test)[:, :10]
        y_pred_label = np.argmax(y_pred, axis=-1)
        f1 = f1_score(self.y_test_label, y_pred_label, average="macro")
        #print(f1)
        self.f1_log.append(f1)

def train(inbalance_size):
    (X_train, y_train), (X_test, y_test) = inbalanced_mnist(inbalance_size)

    model = create_models()
    model.compile("adam", affinity_loss(0.75), [acc])

    scheduler = LearningRateScheduler(step_decay)
    f1 = F1Callback(model, X_test, y_test)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[scheduler, f1],
                        batch_size=128, epochs=100, verbose=0).history

    max_acc = max(history["val_acc"])
    max_f1 = max(f1.f1_log)
    print(f"{inbalance_size} {max_acc:.04} {max_f1:.04}")

if __name__ == "__main__":
    for n in [500, 200, 100, 50, 20, 10]:
        K.clear_session()
        train(n)
