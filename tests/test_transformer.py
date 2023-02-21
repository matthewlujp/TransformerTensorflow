import sys
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from model.transformer import Transformer, get_positional_encoding

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
            self._maxlen_enc, self._maxlen_dec, self._vec_size,
            self._num_stacks, self._num_words)

    def tearDown(self):
        del self._model

    def test_embedding(self):
        x = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        y = self._model.embedding_layer(x)
        self.assertAllEqual(tf.constant([1, self._maxlen_enc, self._vec_size]), tf.shape(y))

    def test_encode_output_shape(self):
        x = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        encoder_output = self._model.encode(x)
        self.assertAllEqual((1, self._maxlen_enc, self._vec_size), encoder_output.shape)

    def test_encode_mask(self):
        l = int(self._maxlen_enc * 0.7)
        x = tf.constant([[1] * l + [0] * (self._maxlen_enc - l)], dtype=tf.float32)
        encoder_output = self._model.encode(x)
        self.assertAllEqual(
            encoder_output[0, l:],
            tf.tile(tf.constant([0] * (self._maxlen_enc - l))[..., None], [1, self._vec_size]))

    def test_decode_output_shape(self):
        enc_input = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        enc_output = tf.ones((1, self._maxlen_enc, self._vec_size), dtype=tf.float32)
        dec_input = tf.ones((1, self._maxlen_dec), dtype=tf.int32)
        dec_output = self._model.decode(enc_input, enc_output, dec_input)
        expected_shape = (1, self._maxlen_dec, self._num_words)
        self.assertAllEqual(dec_output.shape, expected_shape)

    def test_decode_mask(self):
        l = int(self._maxlen_dec * 0.7)
        dec_input = tf.constant([[1] * l + [0] * (self._maxlen_dec - l)], dtype=tf.float32)
        enc_input = tf.ones((1, self._maxlen_enc), dtype=tf.int32)
        enc_output = tf.ones((1, self._maxlen_enc, self._vec_size), dtype=tf.float32)
        dec_output = self._model.decode(enc_input, enc_output, dec_input)
        self.assertAllEqual(
            dec_output[0, l:], 
            tf.zeros((self._maxlen_dec - l, self._num_words), tf.float32))

    def test_positional_encoding(self):
        vec_size = 3
        pos_enc = get_positional_encoding(self._maxlen_enc, vec_size)
        expected = tf.stack([
            tf.math.sin(tf.range(self._maxlen_enc, dtype=tf.float32) / tf.pow(10000.0, 0. / vec_size)),
            tf.math.cos(tf.range(self._maxlen_enc, dtype=tf.float32) / tf.pow(10000.0, 0. / vec_size)),
            tf.math.sin(tf.range(self._maxlen_enc, dtype=tf.float32) / tf.pow(10000.0, 2. / vec_size)),
        ], axis=1)
        self.assertNDArrayNear(pos_enc, expected, 1.0e-5)



if __name__ == "__main__":
    tf.test.main()