import sys
import pathlib
import pickle
from absl import app, flags
import numpy as np
from pyknp import Juman

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from model.transformer import Transformer
from utils import load_data, create_dataset, create_model



FLAGS = flags.FLAGS
flags.DEFINE_string("model_filepath", None, "path to model file", short_name='m')
flags.DEFINE_string("data_filepath", None, "path to data file", short_name='d')
flags.DEFINE_integer("maxlen_enc", 50, "max words in an input prompt")
flags.DEFINE_integer("maxlen_dec", 50, "max words in a model output")
flags.DEFINE_integer("vec_size", 512, "internal vector size")
flags.DEFINE_integer("num_stacks", 6, "number of modules to stack in encoder and decoder")
flags.DEFINE_bool("test", False, "trigger test", short_name='t')
flags.DEFINE_bool("verbose", False, "print process logs")


def verbose(*args):
    if not FLAGS.verbose: return
    print(*args)


def convert_symbols(enc_maxlen, words, word2index, cns_input) -> np.ndarray:
    """
    return: (np.ndarray) 1, L
    """
    # Use Juman++ in subprocess mode
    jumanpp = Juman()
    result = jumanpp.analysis(cns_input)
    input_text=[]
    for mrph in result.mrph_list():
        input_text.append(mrph.midasi)

    #入力データe_inputに入力文の単語インデックスを設定
    e_input=np.zeros((1, enc_maxlen))
    for i, w in enumerate(input_text):
        if i >= enc_maxlen: break
        e_input[0,i] = word2index[w] if w in words else word2index['UNK']

    return e_input



def calculate_response(model: Transformer, words, word2index, index2word, prompt):
    encoder_input = convert_symbols(model.max_encoder_length, words, word2index, prompt)
    encoder_output = model.encode(encoder_input)

    verbose(f"encoder outputs, {[v.shape for v in encoder_output]}")

    decoder_input = np.zeros((1, model.max_decoder_length), np.int32)
    decoder_input[0, 0] = word2index['SSSS']
    decoded_sentence = []

    for i in range(model.max_decoder_length):
        decoder_output = model.decode(
            encoder_input, encoder_output, decoder_input).numpy() # 1, L, NUM_WORDS
        sampled_token = np.argmax(decoder_output[0, i, :], axis=-1)
        sampled_word = index2word[sampled_token]
        verbose(f"sampled token: {sampled_token}   sampled word: {sampled_word}")
        if sampled_word == 'SSSS':
            break

        decoded_sentence.append(sampled_word)
        if i < model.max_decoder_length - 1:
            decoder_input[0, i+1] = sampled_token
    return ''.join(decoded_sentence)



def load_dict(data_filepath):
    data = pickle.loads(pathlib.Path(data_filepath).read_bytes())
    return data['words'], data['word2index'], data['index2word']



def test():
    data = load_data(FLAGS.data_filepath)
    words = data['words']
    word2index = data['word2index']
    index2word = data['index2word']

    train_dataset, _ = create_dataset(data, 0.1)
    model = create_model(FLAGS.maxlen_enc, FLAGS.maxlen_dec, FLAGS.vec_size, FLAGS.num_stacks, len(data['words']), train_dataset)

    prompt = "吾輩は猫である。"
    response = calculate_response(model, words, word2index, index2word, prompt)
    print(f"OUTPUT: {response}")



def chat():
    data = load_data(FLAGS.data_filepath)
    words = data['words']
    word2index = data['word2index']
    index2word = data['index2word']

    train_dataset, _ = create_dataset(data, 0.1)
    model = create_model(FLAGS.maxlen_enc, FLAGS.maxlen_dec, FLAGS.vec_size, FLAGS.num_stacks, len(data['words']), train_dataset)

    while True:
        prompt = input("INPUT > ")
        if prompt in ['q', 'Q', 'quit', 'Quit']:
            break

        response = calculate_response(model, words, word2index, index2word, prompt)
        print(f"OUTPUT: {response}")


def main(_):
    if FLAGS.test:
        test()
    else:
        chat()



if __name__ == '__main__':
    app.run(main)