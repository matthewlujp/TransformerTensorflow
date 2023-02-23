import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl



class MultiheadAttention(tf.keras.Model):

    def __init__(self, vec_size, num_heads, dropout_rate, apply_triangle_mask=False):
        super().__init__()
        assert vec_size % num_heads == 0, f"vec_size ={vec_size} is not multiple of num_head ={num_heads}"
        self._vec_size = vec_size
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate
        self._apply_triangle_mask = apply_triangle_mask

        self._q_dense = tfkl.Dense(self._vec_size, use_bias=False, name="q_dense_layer")
        self._k_dense = tfkl.Dense(self._vec_size, use_bias=False, name="k_dense_layer")
        self._v_dense = tfkl.Dense(self._vec_size, use_bias=False, name="v_dense_layer")
        self._out_dense = tfkl.Dense(self._vec_size, use_bias=False, name="multihead_attention_out_dense_layer")
        self._softmax = tfkl.Softmax()
        self._softmax_dropout = tfkl.Dropout(self._dropout_rate)
        self._out_dropout = tfkl.Dropout(self._dropout_rate)
        self._layer_norm = tfkl.LayerNormalization()

    def call(self, query, memory, memory_mask, training=None):
        """
        query: tf.Tensor B,L,V
        memory: tf.Tensor B,L,V
        memory_mask: tf.Tensor B,L
        """
        Qs = tf.split(self._q_dense(query), self._num_heads, axis=-1) # B, L, h
        Ks = tf.split(self._k_dense(memory), self._num_heads, axis=-1) # B, L, h
        Vs = tf.split(self._v_dense(memory), self._num_heads, axis=-1) # B, L, h

        head_outputs = []
        for h in range(self._num_heads):
            attention = self.calculate_attention(Qs[h], Ks[h], memory_mask) # B, L, L = B, Q, V
            attention = self._softmax_dropout(attention, training=training)
            head = self.weighted_combination(attention, Vs[h])
            head_outputs.append(head)

        x = tf.concat(head_outputs, -1) # B, L, V
        x = self._out_dense(x)
        x = self._out_dropout(x, training=training)
        return self._layer_norm(x + query)

    def calculate_attention(self, query, key, value_mask) -> tf.Tensor:
        attn = query @ tf.transpose(key, [0,2,1]) / tf.sqrt(float(tf.shape(key)[-1])) # B, L, L = B, Q, V

        # masking
        if self._apply_triangle_mask:
            tri_mask = get_triangle_mask(attn) # [[1,0,0,...], [1,1,0,...],...]
            tri_mask = (tri_mask - 1.0) * tf.float32.max
            attn += tri_mask
        attn += (value_mask[:, None, :] - 1.0) * tf.float32.max

        attn = self._softmax(attn) # B, L, L = B, Q, V
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