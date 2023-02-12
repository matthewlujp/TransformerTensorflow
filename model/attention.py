import tensorflow as tf
import tensorflow.keras.layers as tfkl



class MultiheadAttention(tf.keras.Model):

    def __init__(self, vec_size, num_heads, dropout_rate, apply_triangle_mask=False):
        assert vec_size % num_heads == 0, f"vec_size ={vec_size} is not multiple of num_head ={num_head}"
        self._vec_size = vec_size
        self._num_heads = num_heads
        self._head_size = int(vec_size / num_heads)
        self._dropout_rate = dropout_rate
        self._apply_triangle_mask = apply_triangle_mask

    def call(self, query, key, value, query_mask, value_mask):
        """
        query: tf.Tensor B,L,V
        key: tf.Tensor B,L,V
        value: tf.Tensor B,L,V
        query_mask: tf.Tensor B,L
        value_mask: tf.Tensor B,L
        """
        head_outputs = []
        for h in range(self._num_heads):
            Q = tfkl.Dense(self._head_size, use_bias=False)(query) # B, L, h
            K = tfkl.Dense(self._head_size, use_bias=False)(key) # B, L, h
            V = tfkl.Dense(self._head_size, use_bias=False)(value) # B, L, h

            attention = tfkl.Softmax(Q @ tf.transpose(K, [0,2,1]) / tf.sqrt(self._head_size), axis=-1) # B, L, L = B, Q, V

            # masking
            if self._apply_triangle_mask:
                attention *= get_trainagle_mask(attention)
            attention *= query_mask[..., None] * value_mask[:, None, :]

            head = attention @ V
            head_outputs.append(head)

        x = tf.concatenate(head_outputs, axis=-1) # B, L, V
        x = tfkl.Dropout(self._dropout_rate)(x)
        return tf.tfkl.LayerNormalization()(x + query)




def get_trainagle_mask(mat: tf.Tensor) -> tf.Tensor:
    """
    mat: (tf.Tensor) B,L,L

    query x available
    1 0 0 0 ...
    1 1 0 0 ...
    1 1 1 0 ...
    """
    t = tf.ones_like(mat) # B, L, L
    t = tf.cumsum(t, axis=1) # B, L, L
    return tf.cast((t - tf.transpose(t, [0, 2, 1])) >= 0, tf.float32)