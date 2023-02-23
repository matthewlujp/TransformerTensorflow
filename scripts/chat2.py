import pathlib
from absl import app, flags
import numpy as np
import tensorflow as tf

import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("model_filepath", None, "path to model file", short_name='m')
flags.DEFINE_string("data_filepath", None, "path to data file", short_name='d')
flags.DEFINE_integer("maxlen_enc", 50, "max words in an input prompt")
flags.DEFINE_integer("maxlen_dec", 50, "max words in a model output")
flags.DEFINE_integer("vec_size", 512, "internal vector size")
flags.DEFINE_integer("num_stacks", 6, "number of modules to stack in encoder and decoder")


tf.config.run_functions_eagerly(True)


def main(_):
    data = utils.load_data(FLAGS.data_filepath)
    dataset, _ = utils.create_dataset(data, 0.1)
    model = utils.create_model(
        FLAGS.maxlen_enc, FLAGS.maxlen_dec, FLAGS.vec_size, FLAGS.num_stacks, len(data['words']), dataset)
    model.load_weights(FLAGS.model_filepath)

    while True:
        prompt = input("INPUT >")
        
        e_input = utils.convert_symbols(50, data['words'], data['word2index'], prompt)
        e_output = model.encode(e_input)

        d_input = np.zeros((1, 50), dtype=np.int32)
        d_input[0, 0] = data['word2index']['SSSS']

        response = []
        
        for i in range(50):
            d_output = model.decode(e_input, e_output, d_input).numpy()

            output_idx = tf.argmax(d_output[0,i], axis=-1).numpy()
            w = data['index2word'][output_idx]
            if w == 'SSSS':
                break
            response.append(w)
            # print(w)
            if i+1 < 50:
                d_input[0, i+1] = int(output_idx)

        print("OUTPUT:", ''.join(response))


if __name__ == "__main__":
    app.run(main)    