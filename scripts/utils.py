import sys
import pathlib
import pickle
import numpy as np
import tensorflow as tf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from model.transformer import Transformer


def load_data(filepath) -> dict:
    data = pickle.loads(pathlib.Path(filepath).read_bytes())
    return data


def create_dataset(data, val_split_rate: float) -> tuple:
    data_size = data['encoder_inputs'].shape[0]
    val_size = int(np.round(data_size * val_split_rate))
    print(
        f"""\n================================================================================

        Total data size: {data_size}  (train: {data_size - val_size},   val: {val_size})

        ================================================================================\n""")

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'encoder_input': data['encoder_inputs'][val_size:], 'decoder_input': data['decoder_inputs'][val_size:]},
        data['labels'][val_size:]))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'encoder_input': data['encoder_inputs'][:val_size], 'decoder_input': data['decoder_inputs'][:val_size]},
        data['labels'][:val_size]))

    return train_dataset, val_dataset
    


def create_model(maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words, dataset) -> tf.keras.Model:
    model = Transformer(maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words)
    x, _ = next(iter(dataset.batch(10)))
    _ = model(x)
    return model
