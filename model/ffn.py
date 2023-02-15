import tensorflow as tf
import tensorflow.keras.layers as tfkl


class FFN(tf.keras.Model):

    def __init__(self, maxlen, vec_size, dropout_rate):
        super().__init__()
        self._maxlen = maxlen
        self._vec_size = vec_size
        self._dropout_rate = dropout_rate

        self._hidden_dense_layers = [tfkl.Dense(self._vec_size * 4, activation=tf.nn.relu) for _ in range(self._maxlen)]
        self._out_dense_layers = [tfkl.Dense(self._vec_size) for _ in range(self._maxlen)]
        self._dropout = tfkl.Dropout(self._dropout_rate)
        self._layer_norm = tfkl.LayerNormalization()

    def call(self, x):
        """
        x: tf.Tensor B, L, V
        """
        # positional fn
        position_outputs = []
        for i in range(self._maxlen):
            h = tf.gather(x, i, axis=1)
            h = self._hidden_dense_layers[i](h)
            h = self._out_dense_layers[i](h)
            position_outputs.append(h)

        residual = tf.stack(position_outputs, axis=1) # [(B, V), (B, V), ...] -> B, L, V
        residual = self._dropout(residual)
        return self._layer_norm(residual + x)


        



        