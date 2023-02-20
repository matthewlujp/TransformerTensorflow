import sys
import pathlib 
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from train import masked_sparse_entropy, masked_perplexity, masked_accuracy


np.random.seed(0)
random_seed.set_seed(42)


class TestMetrics(tf.test.TestCase):

    def setUp(self):
        self._vec_size = 16
        self._num_words = 5
        self._maxlen_dec = 3

    def test_masked_sparse_entropy_all_match(self):
        y_true = tf.constant([[1, 2, 3]], dtype=tf.int32)
        y_pred = tf.constant([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ], dtype=tf.float32)
        expected = masked_sparse_entropy(y_true, y_pred)
        softmax = tf.nn.softmax(y_pred, axis=-1) # 1,3,5
        expected = (
            - tf.math.log(softmax[0, 0, 1])
            - tf.math.log(softmax[0, 1, 2])
            - tf.math.log(softmax[0, 2, 3])) / 3
        self.assertEqual(expected, expected)

    def test_masked_sparse_entropy_partial_match(self):
        y_true = tf.constant([[1, 3, 4]], dtype=tf.int32)
        y_pred = tf.constant([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ], dtype=tf.float32)
        expected = masked_sparse_entropy(y_true, y_pred)
        softmax = tf.nn.softmax(y_pred, axis=-1) # 1,3,5
        expected = (
            - tf.math.log(softmax[0, 0, 1])
            - tf.math.log(softmax[0, 1, 3])
            - tf.math.log(softmax[0, 2, 4])) / 3
        self.assertEqual(expected, expected)

    def test_masked_sparse_entropy_mased(self):
        y_true = tf.constant([[1, 2, 0]], dtype=tf.int32)
        y_pred = tf.constant([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ], dtype=tf.float32)
        expected = masked_sparse_entropy(y_true, y_pred)
        softmax = tf.nn.softmax(y_pred, axis=-1) # 1,3,5
        expected = (
            - tf.math.log(softmax[0, 0, 1])
            - tf.math.log(softmax[0, 1, 2])) / 2
        self.assertEqual(expected, expected)

    def test_masked_accuracy_masked(self):
        y_true = tf.constant([[1, 3, 0]], dtype=tf.float32)
        y_pred = tf.constant([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ], dtype=tf.float32)
        acc = masked_accuracy(y_true, y_pred)
        expected = 1/2
        self.assertEqual(acc, expected)

    def test_masked_accuracy_partial_match(self):
        y_true = tf.constant([[1, 3, 3]], dtype=tf.float32)
        y_pred = tf.constant([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ], dtype=tf.float32)
        acc = masked_accuracy(y_true, y_pred)
        expected = 2/3
        self.assertEqual(acc, expected)



if __name__ == "__main__":
    tf.test.main()