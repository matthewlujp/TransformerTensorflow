import pathlib
import pickle
import numpy as np
from absl import app, flags


def create_dict(conversations_words_filepath):
    conversations_words_filepath = pathlib.Path(conversations_words_filepath)
    parts_list = pickle.loads(conversations_words_filepath.read_bytes())
    words = sorted(list(set(parts_list)))
    print(f"total words: {len(words)}")

    word2index = {w: i for i, w in enumerate(words)}
    index2word = {i: w for i, w in enumerate(words)}

    cnt = np.zeros(len(words), dtype='int32')
    for w in parts_list:
        cnt[word2index[w]] += 1
    
    # eliminate less frequent words
    for k in range(len(words)):
        if cnt[k] <= 3: words[k] = 'UNK'

    words = sorted(list(set(words)) + ['\t'])
    print(f"new total words: {len(words)}")
    
    word2index = {w: i for i, w in enumerate(words)}
    index2word = {i: w for i, w in enumerate(words)}

    list_urtext = [
        word2index[w] if w in words else word2index['UNK']
        for w in parts_list]
    return words, word2index, index2word, np.array(list_urtext)


def create_training_data(maxlen_e, maxlen_d, words, word2index, index2word, urtext):
    # convert to list sentences
    # a sentence is an array of word indices 
    # data = [np.array([x, x, x, ...]), ...]
    separator = word2index['SSSS']
    data = []
    for i, word_idx in enumerate(urtext[:-1]):
        row = [separator] if word_idx == separator else row + [word_idx]

        if urtext[i + 1] == separator and len(row) > 1:
            data.append(np.array(row))
        elif i == len(urtext) - 2:
            row.append(urtext[i+1])
            data.append(np.array(row))

    print(len(data))

    # convert to training data
    e_inputs = []
    d_inputs = []
    t_labels = []
    for i in range(len(data) - 1):
        # do not include 'UNK' in decoder inputs and labels
        if np.any(data[i+1]) == word2index['UNK'] or len(data[i+1]) >= maxlen_d + 1:
            continue

        e_row = np.zeros((maxlen_e,), dtype='int32')
        d_row = np.zeros((maxlen_d,), dtype='int32')
        t_row = np.zeros((maxlen_d,), dtype='int32')
        e_data = data[i]
        d_data = data[i+1]
        e_end = min(len(e_data), maxlen_e + 1)
        d_end = min(len(d_data), maxlen_d)
        t_end = min(len(d_data), maxlen_d + 1)
        e_row[:e_end-1] = e_data[1:e_end]
        d_row[:d_end] = d_data[:d_end]
        t_row[:t_end-1] = d_data[1:d_end]
        if t_end <= maxlen_d:
            t_row[t_end - 1] = separator
        e_inputs.append(e_row)
        d_inputs.append(d_row)
        t_labels.append(t_row)

    # shuffle
    z = list(zip(e_inputs, d_inputs, t_labels))
    np.random.seed(0)
    np.random.shuffle(z)
    e, d, t = zip(*z)

    e = np.array(e).reshape(len(e_inputs), maxlen_e)
    d = np.array(d).reshape(len(d_inputs), maxlen_d)
    t = np.array(t).reshape(len(t_labels), maxlen_d)
    print(f"encoder input: {e.shape}, decoder input: {d.shape}, labels: {t.shape}")

    return e, d, t



FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_save_file", None, "relative path to save created dataset", short_name='s')
flags.DEFINE_string("datafile", None, "relative path of word sperated sentence file data", short_name='f')
flags.DEFINE_integer("maxlen_encoder", 50, "max length for encoder input", short_name='mx_e')
flags.DEFINE_integer("maxlen_decoder", 50, "max length for decoder input", short_name='mx_d')


def main(_):
    data_dir_filepath = pathlib.Path(__file__).resolve().parents[1] / 'data'
    words, word2index, index2word, urtext = create_dict(data_dir_filepath / FLAGS.datafile)
    e, d, t = create_training_data(FLAGS.maxlen_encoder, FLAGS.maxlen_decoder, words, word2index, index2word, urtext)

    (data_dir_filepath / FLAGS.dataset_save_file).write_bytes(pickle.dumps(
        {
            'word2index': word2index,
            'index2word': index2word,
            'words': words,
            'maxlen_encoder': FLAGS.maxlen_encoder,
            'maxlen_decoder': FLAGS.maxlen_decoder,
            'encoder_inputs': e,
            'decoder_inputs': d,
            'labels': t,
        }
    ))

    # save_dir_path = data_dir_filepath / FLAGS.dataset_dir
    # if not save_dir_path.exists():
    #     save_dir_path.mkdir(parents=True)
    # (save_dir_path / 'word2index.pickle').write_bytes(pickle.dumps(word2index))
    # (save_dir_path / 'index2word.pickle').write_bytes(pickle.dumps(index2word))
    # (save_dir_path / 'words.pickle').write_bytes(pickle.dumps(words))
    # (save_dir_path / 'encoder_inputs.pickle').write_bytes(pickle.dumps(e))
    # (save_dir_path / 'decoder_inputs.pickle').write_bytes(pickle.dumps(d))
    # (save_dir_path / 'labels.pickle').write_bytes(pickle.dumps(t))
    # (save_dir_path / 'maxlen.pickle').write_bytes(pickle.dumps({'maxlen_e': FLAGS.maxlen_encoder, 'maxlen_d': FLAGS.maxlen_decoder}))



if __name__ == '__main__':
    app.run(main)







    




    
     

