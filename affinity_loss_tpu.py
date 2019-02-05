import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class ClusteringAffinity(layers.Layer):
    def __init__(self, n_classes, n_centers, sigma, **kwargs):
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.sigma = sigma
        super().__init__(**kwargs)

    def build(self, input_shape):
        # batch_size = N, n_classes = C, n_latent_dims = h, centers = m
        # center(W) = (C, m, h)
        # input(f) = (N, h)
        self.W = self.add_weight(name="W", shape=(self.n_classes, self.n_centers, int(input_shape[1])),
                                 initializer="he_normal", trainable=True)
        super().build(input_shape)

    # upper triangular matrix without diagonal
    def upper_triangle(self, matrix):
        upper = tf.matrix_band_part(matrix, 0, -1)
        diagonal = tf.matrix_band_part(matrix, 0, 0)
        diagonal_mask = tf.sign(tf.abs(tf.matrix_band_part(diagonal, 0, 0)))
        return upper * (1.0 - diagonal_mask)

    def call(self, f):
        # Euclidean space similarity measure
        # calculate d(f_i, w_j)
        f_expand = tf.expand_dims(tf.expand_dims(f, axis=1), axis=1)
        w_expand = tf.expand_dims(self.W, axis=0)
        fw_norm = tf.reduce_sum((f_expand-w_expand)**2, axis=-1)
        distance = tf.exp(-fw_norm/self.sigma)
        distance = tf.reduce_max(distance, axis=-1) # (N,C,m)->(N,C)

        # Regularization
        hidden_layers = K.int_shape(self.W)[2]
        mc = self.n_classes * self.n_centers
        w_reshape = tf.reshape(self.W, [mc, hidden_layers])
        w_reshape_expand1 = tf.expand_dims(w_reshape, axis=0)
        w_reshape_expand2 = tf.expand_dims(w_reshape, axis=1)
        w_norm_mat = tf.reduce_sum((w_reshape_expand2 - w_reshape_expand1)**2, axis=-1)
        w_norm_upper = self.upper_triangle(w_norm_mat)
        mu = 2.0 / (mc**2 - mc) * tf.reduce_sum(w_norm_upper)
        residuals = self.upper_triangle((w_norm_upper - mu)**2)
        rw = 2.0 / (mc**2 - mc) * tf.reduce_sum(residuals)

        batch_size = tf.shape(f)[0]
        rw_broadcast = tf.ones((batch_size,1)) * rw

        # outputs distance(N, C) + rw(N,)
        output = tf.concat([distance, rw_broadcast], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes+1)

    def get_config(self):
        config = {
            "n_classes":self.n_classes,
            "n_centers":self.n_centers,
            "sigma":self.sigma
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def affinity_loss(lambd):
    def loss(y_true_plusone, y_pred_plusone):
        # true
        onehot = y_true_plusone[:, :-1]
        # pred
        distance = y_pred_plusone[:, :-1]
        rw = tf.reduce_mean(y_pred_plusone[:, -1])

        # L_mm
        d_fi_wyi = tf.reduce_sum(onehot * distance, axis=-1, keepdims=True)
        losses = tf.maximum(lambd + distance - d_fi_wyi, 0.0)
        L_mm = tf.reduce_sum(losses * (1.0-onehot), axis=-1) # j!=y_i

        return L_mm + rw
    return loss
