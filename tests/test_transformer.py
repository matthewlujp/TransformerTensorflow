import sys
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

sys.path.append(pathlib.Path(__file__).resolve().parents[1])
from model.transformer import Transformer

np.random.seed(0)
random_seed.set_seed(42)


class TestTransformer(tf.test.TestCase):

    def setUp(self):
        super().setUp()

        self._maxlen_enc = 10
        self._maxlen_dec = 10
        self._vec_size = 24
        self._num_stacks = 3
        self._num_words = 200

        self._model = Transformer(
            self._maxlen_enc, self._maxlen_dec, self._vec_size, self._num_stacks, self._num_words)
        
    def tearDown(self):
        del self._model

    def test_embedding(self):
        x = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        y = self._model.embedding_layer(x)
        self.assertAllEqual(tf.constant([1, self._maxlen_enc, self._vec_size]), tf.shape(y))

    def test_encode_output_shape(self):
        x = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        encoder_outputs = self._model.encode(x)
        self.assertEqual(len(encoder_outputs), self._num_stacks)
        for o in encoder_outputs:
            self.assertAllEqual((1, self._maxlen_enc, self._vec_size), o.shape)

    def test_encode_mask(self):
        l = int(self._maxlen_enc * 0.7)
        x = tf.constant([[1] * l + [0] * (self._maxlen_enc - l)], dtype=tf.float32)
        encoder_outputs = self._model.encode(x)
        for i, o in enumerate(encoder_outputs):
            self.assertAllEqual(
                o[0, l:],
                tf.tile(tf.constant([0] * (self._maxlen_enc - l))[..., None], [1, self._vec_size]),
                f"layer {i}, result {o}")

    def test_decode_output_shape(self):
        enc_input = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        enc_outputs = [tf.ones((1, self._maxlen_enc, self._vec_size), dtype=tf.float32) for _ in range(self._num_stacks)]
        dec_input = tf.ones((1, self._maxlen_dec), dtype=tf.int32)
        dec_output = self._model.decode(enc_input, enc_outputs, dec_input)
        expected_shape = (1, self._maxlen_dec, self._num_words)
        self.assertAllEqual(dec_output.shape, expected_shape)

    def test_decode_mask(self):
        l = int(self._maxlen_dec * 0.7)
        dec_input = tf.constant([[1] * l + [0] * (self._maxlen_dec - l)], dtype=tf.float32)
        enc_input = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        enc_outputs = [tf.ones((1, self._maxlen_enc, self._vec_size), dtype=tf.float32) for _ in range(self._num_stacks)]
        dec_output = self._model.decode(enc_input, enc_outputs, dec_input)
        self.assertAllEqual(
            dec_output[0, l:], 
            tf.zeros((self._maxlen_dec - l, self._num_words), tf.float32))

        
if __name__ == "__main__":
    tf.test.main()