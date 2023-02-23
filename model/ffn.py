import tensorflow as tf
import tensorflow.keras.layers as tfkl


class FFN(tf.keras.Model):

    def __init__(self, maxlen, vec_size, dropout_rate):
        super().__init__()
        self._maxlen = maxlen
        self._vec_size = vec_size
        self._dropout_rate = dropout_rate

        self._hidden_dense = tfkl.Dense(self._vec_size * 4, activation=tf.nn.relu)
        self._out_dense = tfkl.Dense(self._vec_size)
        self._hidden_dropout = tfkl.Dropout(self._dropout_rate)
        self._out_dropout = tfkl.Dropout(self._dropout_rate)
        self._layer_norm = tfkl.LayerNormalization()

    def call(self, x, training=None):
        """
        x: tf.Tensor B, L, V
        """
        residual = x
        residual = self._hidden_dense(residual)
        residual = self._hidden_dropout(residual, training=training)
        residual = self._out_dense(residual)
        residual = self._out_dropout(residual, training=training)
        return self._layer_norm(residual + x)


        



        