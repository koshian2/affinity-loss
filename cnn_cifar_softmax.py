import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from tensorflow.contrib.tpu.python.tpu import keras_support

from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from datasets import inbalanced_cifar
from sklearn.metrics import f1_score
import os


def conv_bn_relu(input, ch):
    x = layers.Conv2D(ch, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def create_models():
    input = layers.Input((32,32,3))
    x = input
    for i in range(3):
        x = conv_bn_relu(x, 64)
    x = layers.AveragePooling2D(2)(x)
    for i in range(3):
        x = conv_bn_relu(x, 128)
    x = layers.AveragePooling2D(2)(x)
    for i in range(3):
        x = conv_bn_relu(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    return Model(input, x)


def step_decay(epoch):
    x = 5e-3
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
        self.f1_log.append(f1)

def train(inbalance_size):
    (X_train, y_train), (X_test, y_test) = inbalanced_cifar(inbalance_size)
    y_train, y_test = y_train[:, :10], y_test[:, :10]

    model = create_models()
    model.compile("adam", "categorical_crossentropy", ["acc"])
    tf.logging.set_verbosity(tf.logging.FATAL)

    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    scheduler = LearningRateScheduler(step_decay)
    f1 = F1Callback(model, X_test, y_test)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[scheduler, f1],
                        batch_size=640, epochs=100, verbose=0).history

    max_acc = max(history["val_acc"])
    max_f1 = max(f1.f1_log)
    print(f"{inbalance_size} {max_acc:.04} {max_f1:.04}")

if __name__ == "__main__":
    for i in range(1):
        print("-----", i, "-----")
        for n in [500, 200, 100, 50, 20, 10]:
            K.clear_session()
            train(n)
