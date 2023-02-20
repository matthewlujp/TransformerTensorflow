import sys
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

sys.path.append(pathlib.Path(__file__).resolve().parents[1])
from model.ffn import FFN

np.random.seed(0)
random_seed.set_seed(42)


class TestFFN(tf.test.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self._vec_size = 512
        self._maxlen = 50
        self._ffn = FFN(self._maxlen, self._vec_size, 0.1)

    def tearDown(self):
        del self._ffn

    def test_ffn_output_shape(self):
        x = tf.zeros((1, self._maxlen, self._vec_size), dtype=tf.float32)
        y = self._ffn(x)
        self.assertShapeEqual(x, y)



if __name__ == "__main__":
    tf.test.main()