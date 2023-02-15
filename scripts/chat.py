import sys
import pathlib
import pickle
from absl import app, flags
import numpy 
import tensorflow

sys.path.append(pathlib.Path(__file__).resolve().parents[1] / 'model')

from model.transformer import Transformer


FLAGS = flags.FLAGS
FLAGS.DEFINE_string("model_filepath", None, "path to model file", short_name='m')




def calculate_response(model, prompt):
    
    pass


def load_dict(data_filepath):
    data = pickle.loads(pathlib.Path(data_filepath).read_bytes())
    return data['words'], data['word2index'], data['index2word']



def chat(_):
    
    model = tf.keras.models.load_model(FLAGS.model_filepath)

    while True:
        prompt = input("INPUT > ")
        if prompt in ['q', 'Q', 'quit', 'Quit']:
            break

        response = calculate_response(model, prompt)
        print(f"OUTPUT: {response}")



if __name__ == '__main__':
    app.run(chat)