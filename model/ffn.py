import tensorflow as tf
import tensorflow.keras.layers as tfkl


class FFN(tf.keras.Model):

    def __init__(self, maxlen, vec_size, dropout_rate):
        self._maxlen = maxlen
        self._vec_size = vec_size
        self._dropout_rate = dropout_rate

    def call(self, x):
        """
        x: tf.Tensor B, L, V
        """
        # positional fn
        position_outputs = []
        for i in range(self._maxlen):
            h = tf.gather(x, i, axis=1)
            h = tfkl.Dense(self._vec_size * 4, activation=tf.nn.relu)(h)
            h = tfkl.Dense(self._vec_size)(h)
            position_outputs.append(h)

        residual = tf.stack(position_outputs, axis=1) # [(B, V), (B, V), ...] -> B, L, V
        residual = tfkl.Dropout(self._dropout_rate)(residual)
        return tfkl.LayerNormalization()(residual + x)


        



        