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
FLAGS.DEFINE_string("data_filepath", None, "path to data file", short_name='d')


def encode_request(cns_input, data) :
    maxlen_e      = data['maxlen_e']
    maxlen_d      = data['maxlen_d']
    word_indices  = data['word_indices']
    words         = data['words']
    encoder_model = data['encoder_model']

    # Use Juman++ in subprocess mode
    jumanpp = Juman()
    result = jumanpp.analysis(cns_input)
    input_text=[]
    for mrph in result.mrph_list():
        input_text.append(mrph.midasi)

    mat_input=np.array(input_text)

    #入力データe_inputに入力文の単語インデックスを設定
    e_input=np.zeros((1,maxlen_e))
    for i in range(0,len(mat_input)) :
        if mat_input[i] in words :
            e_input[0,i] = word_indices[mat_input[i]]
        else :
            e_input[0,i] = word_indices['UNK']

    return e_input



def calculate_response(model, words, word2index, index2word, prompt):
    
    pass


def load_dict(data_filepath):
    data = pickle.loads(pathlib.Path(data_filepath).read_bytes())
    return data['words'], data['word2index'], data['index2word']



def chat(_):
    dictionary = load_data()    
    model = tf.keras.models.load_model(FLAGS.model_filepath)

    while True:
        prompt = input("INPUT > ")
        if prompt in ['q', 'Q', 'quit', 'Quit']:
            break

        response = calculate_response(model, prompt)
        print(f"OUTPUT: {response}")



if __name__ == '__main__':
    app.run(chat)