import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from tensorflow.contrib.tpu.python.tpu import keras_support
from affinity_loss_tpu import *
from datasets import inbalanced_cifar

from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import os, optuna

def conv_bn_relu(input, ch):
    x = layers.Conv2D(ch, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def create_models(sigma, m):
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
    x = layers.BatchNormalization()(x)
    x = ClusteringAffinity(10, m, sigma)(x)

    return Model(input, x)

def acc(y_true_plusone, y_pred_plusone):
    y_true = K.argmax(y_true_plusone[:, :-1], axis=-1)
    y_pred = K.argmax(y_pred_plusone[:, :-1], axis=-1)
    equals = K.cast(K.equal(y_true, y_pred), "float")
    return K.mean(equals)

def step_decay(epoch):
    x = 5e-3
    if epoch >= 50: x /= 5.0
    if epoch >= 80: x /= 5.0
    return x

class F1Callback(Callback):
    def __init__(self, model, X_test, y_test, trial):
        self.model = model
        self.X_test = X_test
        self.y_test_label = np.argmax(y_test, axis=-1)
        self.trial = trial
        self.f1_log = []

    def on_epoch_end(self, epoch, logs):
        y_pred = self.model.predict(self.X_test)[:, :10]
        y_pred_label = np.argmax(y_pred, axis=-1)
        f1 = f1_score(self.y_test_label, y_pred_label, average="macro")
        self.f1_log.append(f1)
        # optuna prune
        self.trial.report(1.0-f1, step=epoch)
        if self.trial.should_prune(epoch):
            raise optuna.structs.TrialPruned()


def train(lambd, sigma, n_centers, trial):
    K.clear_session()
    (X_train, y_train), (X_test, y_test) = inbalanced_cifar(200)

    model = create_models(sigma, n_centers)
    model.compile("adam", affinity_loss(lambd), [acc])
    tf.logging.set_verbosity(tf.logging.FATAL) # ログを埋めないようにする

    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    scheduler = LearningRateScheduler(step_decay)
    f1 = F1Callback(model, X_test, y_test, trial)

    history = model.fit(X_train, y_train, callbacks=[scheduler, f1],
                        batch_size=640, epochs=100, verbose=0).history

    max_f1 = max(f1.f1_log)
    print(f"lambda:{lambd:.04}, sigma:{sigma:.04} n_centers:{n_centers} / f1 = {max_f1:.04}")
    return max_f1

def objective(trial):
    lambd = trial.suggest_uniform("lambda", 0.01, 1.0)
    sigma = trial.suggest_loguniform("sigma", 10.0, 100.0)
    center = trial.suggest_int("n_centers", 1, 30)
    max_f1 = train(lambd, sigma, center, trial)
    return 1.0-max_f1

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=200)
    df = study.trials_dataframe()
    df.to_csv("optuna_affinity.csv")
