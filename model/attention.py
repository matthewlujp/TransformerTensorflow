import tensorflow as tf
import tensorflow.keras.layers as tfkl



class MultiheadAttention(tf.keras.Model):

    def __init__(self, vec_size, num_heads, dropout_rate, apply_triangle_mask=False):
        super().__init__()
        assert vec_size % num_heads == 0, f"vec_size ={vec_size} is not multiple of num_head ={num_heads}"
        self._vec_size = vec_size
        self._num_heads = num_heads
        self._head_size = int(vec_size / num_heads)
        self._dropout_rate = dropout_rate
        self._apply_triangle_mask = apply_triangle_mask

        self._q_dense_layers = [tfkl.Dense(self._head_size, use_bias=False) for _ in range(self._num_heads)]
        self._k_dense_layers = [tfkl.Dense(self._head_size, use_bias=False) for _ in range(self._num_heads)]
        self._v_dense_layers = [tfkl.Dense(self._head_size, use_bias=False) for _ in range(self._num_heads)]
        self._softmax = tfkl.Softmax()
        self._dropout = tfkl.Dropout(self._dropout_rate)
        self._layer_norm = tfkl.LayerNormalization()

    def call(self, query, key, value, query_mask, value_mask, training=None):
        """
        query: tf.Tensor B,L,V
        key: tf.Tensor B,L,V
        value: tf.Tensor B,L,V
        query_mask: tf.Tensor B,L
        value_mask: tf.Tensor B,L
        """
        head_outputs = []
        for h in range(self._num_heads):
            Q = self._q_dense_layers[h](query) # B, L, h
            K = self._k_dense_layers[h](key) # B, L, h
            V = self._v_dense_layers[h](value) # B, L, h

            attention = self.calculate_attention(Q, K, query_mask, value_mask) # B, L, L = B, Q, V

            head = self.weighted_combination(attention, V)
            head_outputs.append(head)

        x = tf.concat(head_outputs, -1) # B, L, V
        x = self._dropout(x, training=training)
        return self._layer_norm(x + query)

    def calculate_attention(self, query, key, query_mask, value_mask) -> tf.Tensor:
        attn = self._softmax(query @ tf.transpose(key, [0,2,1]) / tf.sqrt(float(self._head_size))) # B, L, L = B, Q, V

        # masking
        if self._apply_triangle_mask:
            attn *= get_triangle_mask(attn)
        attn *= query_mask[..., None] * value_mask[:, None, :]
        return attn

    def weighted_combination(self, weights: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        return weights @ values



def get_triangle_mask(mat: tf.Tensor) -> tf.Tensor:
    """
    mat: (tf.Tensor) B,L,L

    query x available
    1 0 0 0 ...
    1 1 0 0 ...
    1 1 1 0 ...
    """
    return tf.linalg.band_part(tf.ones_like(mat), -1, 0)