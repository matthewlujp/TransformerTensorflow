import sys
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

sys.path.append(pathlib.Path(__file__).resolve().parents[1])
from model.attention import MultiheadAttention, get_triangle_mask

np.random.seed(0)
random_seed.set_seed(42)


class TestAttention(tf.test.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self._vec_size = 512
        self._num_heads = 8

        self._self_attention = MultiheadAttention(
            self._vec_size, self._num_heads, 0.1, apply_triangle_mask=False)
        self._dec_self_attention = MultiheadAttention(
            self._vec_size, self._num_heads, 0.1, apply_triangle_mask=True)

    def tearDown(self) -> None:
        del self._self_attention
        del self._dec_self_attention

    def test_get_triangle_mask(self):
        result = get_triangle_mask(tf.ones((1, 5, 5)))
        expected = tf.constant([
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]],
            dtype=tf.float32)
        self.assertAllEqual(expected, result)

    def test_attention_without_mask(self):
        Q = tf.constant([[[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, -1]]], dtype=tf.float32) # 1, 4, 3
        K = tf.constant([[[1, 0, 1], [1, 1, 0], [1, 0, -1], [0, 1, 1]]], dtype=tf.float32) # 1, 4, 3

        attn = self._self_attention.calculate_attention(
            Q, K, tf.ones((1, 4), dtype=tf.float32), tf.ones((1, 4), dtype=tf.float32))
        expected_attn = tf.constant([
            [
                [1, 1, -1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 1],
                [0, 1, 2, -1],
            ],
        ], dtype=tf.float32)
        expected_attn /= tf.math.sqrt(self._vec_size / self._num_heads)
        expected_attn = tf.keras.layers.Softmax()(expected_attn)
        self.assertAllEqual(expected_attn, attn)

    def test_attention_with_query_mask(self):
        Q = tf.constant([[[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, -1]]], dtype=tf.float32) # 1, 4, 3
        K = tf.constant([[[1, 0, 1], [1, 1, 0], [1, 0, -1], [0, 1, 1]]], dtype=tf.float32) # 1, 4, 3
        query_mask = tf.constant([[1, 1, 0, 0]], dtype=tf.float32)

        attn = self._self_attention.calculate_attention(
            Q, K, query_mask, tf.ones((1, 4), dtype=tf.float32))
        expected_attn = tf.constant([
            [
                [1, 1, -1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 1],
                [0, 1, 2, -1],
            ],
        ], dtype=tf.float32)
        expected_attn /= tf.math.sqrt(self._vec_size / self._num_heads)
        expected_attn = tf.keras.layers.Softmax()(expected_attn)
        expected_attn *= tf.constant([
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ], dtype=tf.float32)
        self.assertAllEqual(expected_attn, attn)

    def test_attention_with_value_mask(self):
        Q = tf.constant([[[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, -1]]], dtype=tf.float32) # 1, 4, 3
        K = tf.constant([[[1, 0, 1], [1, 1, 0], [1, 0, -1], [0, 1, 1]]], dtype=tf.float32) # 1, 4, 3
        value_mask = tf.constant([[1, 1, 1, 0]], dtype=tf.float32)

        attn = self._self_attention.calculate_attention(
            Q, K, tf.ones((1, 4), dtype=tf.float32), value_mask)
        expected_attn = tf.constant([
            [
                [1, 1, -1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 1],
                [0, 1, 2, -1],
            ],
        ], dtype=tf.float32)
        expected_attn /= tf.math.sqrt(self._vec_size / self._num_heads)
        expected_attn = tf.keras.layers.Softmax()(expected_attn)
        expected_attn *= tf.constant([
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
        ], dtype=tf.float32)
        self.assertAllEqual(expected_attn, attn)

    def test_decoder_self_attention(self):
        Q = tf.constant([[[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, -1]]], dtype=tf.float32) # 1, 4, 3
        K = tf.constant([[[1, 0, 1], [1, 1, 0], [1, 0, -1], [0, 1, 1]]], dtype=tf.float32) # 1, 4, 3
        query_mask = tf.constant([[1, 1, 1, 0]], dtype=tf.float32)

        attn = self._dec_self_attention.calculate_attention(
            Q, K, query_mask, tf.ones((1, 4), dtype=tf.float32))
        expected_attn = tf.constant([
            [
                [1, 1, -1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 1],
                [0, 1, 2, -1],
            ],
        ], dtype=tf.float32)
        expected_attn /= tf.math.sqrt(self._vec_size / self._num_heads)
        expected_attn = tf.keras.layers.Softmax()(expected_attn)
        expected_attn *= tf.constant([
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
        ], dtype=tf.float32)
        self.assertAllEqual(expected_attn, attn)

    def test_weigted_combination(self):
        weights = tf.constant([
            [
                [0.5, 0, 0, 0.5],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]], dtype=tf.float32)
        V = tf.constant([[[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]]], dtype=tf.float32) # 1, 4, 3

        result = self._self_attention.weighted_combination(weights, V)
        expected = tf.constant([
            [
                [25, 25, 25],
                [20, 20, 20],
                [30, 30, 30],
                [40, 40, 40],

            ]], dtype=tf.float32)
        self.assertAllEqual(expected, result)



if __name__ == "__main__":
    tf.test.main()
