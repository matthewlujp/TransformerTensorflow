import sys
import pathlib
import pickle
from absl import app, flags
import numpy as np
import tensorflow as tf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from model.transformer import Transformer


FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 100, "number of epochs", short_name='e')
flags.DEFINE_integer("batch_size", 16, "batch size", short_name='b')
flags.DEFINE_string("checkpoint_dir", None, "path to directory where checkpoints are saved", short_name='ckdir')
flags.DEFINE_string("logdir", None, "path to directory for TensorBoard", short_name='l')
flags.DEFINE_string("final_model_filepath", None, "path to save the final model", short_name='f')
flags.DEFINE_string("data_filepath", None, "path to data", short_name='d')
flags.DEFINE_float("validation_split_rate", 0.2, "ration of validation data in total", short_name='s')
flags.DEFINE_integer("maxlen_enc", 50, "max words in an input prompt")
flags.DEFINE_integer("maxlen_dec", 50, "max words in a model output")
flags.DEFINE_integer("vec_size", 512, "internal vector size")
flags.DEFINE_integer("num_stacks", 6, "number of modules to stack in encoder and decoder")
flags.DEFINE_integer("checkpoint_every", 10, "number of epochs to save checkpoint")





def masked_sparse_entropy(y_true, y_pred):
    """
    y_true: B, L
    y_pred: B, L, NUM_WORDS
    """
    mask = tf.cast(tf.sign(tf.abs(y_true)), dtype=tf.float32) # B, L  (1 if word, 0 if padding)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred) # B, L
    loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
    return tf.reduce_mean(loss)    


def masked_perplexity(y_true, y_pred):
    """
    y_true: B, L
    y_pred: B, L, NUM_WORDS
    """
    mask = tf.cast(tf.sign(tf.abs(y_true)), dtype=tf.float32) # B, L  (1 if word, 0 if padding)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred) # B, L
    loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
    return tf.reduce_mean(tf.exp(loss))


def masked_accuracy(y_true, y_pred):
    """
    y_true: B, L
    y_pred: B, L, NUM_WORDS
    """
    mask = tf.cast(tf.sign(tf.abs(y_true)), dtype=tf.float32) # B, L  (1 if word, 0 if padding)
    sentense_len = tf.reduce_sum(mask, axis=-1) # B
    acc = tf.cast(tf.equal(y_true, tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)), dtype=tf.float32) # B, L
    masked_acc = acc * mask # B, L
    return tf.reduce_mean(masked_acc)
    


def load_data(filepath):
    data = pickle.loads(pathlib.Path(filepath).read_bytes())
    return data


def create_dataset(data, val_split_rate):
    val_size = int(np.round(data['encoder_inputs'].shape[0] * val_split_rate))
    train_size = data['encoder_inputs'].shape[0] - val_size

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'encoder_input': data['encoder_inputs'][:val_size], 'decoder_input': data['decoder_inputs'][:val_size]},
        data['labels'][:val_size]))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'encoder_input': data['encoder_inputs'][val_size:], 'decoder_input': data['decoder_inputs'][val_size:]},
        data['labels'][val_size:]))
    return train_dataset, val_dataset
    


def create_model(maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words, dataset):
    model = Transformer(maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words)
    x, _ = next(iter(dataset.batch(10)))
    _ = model(x)
    return model
    




def train(_):
    data = load_data(FLAGS.data_filepath)
    train_dataset, val_dataset = create_dataset(data, FLAGS.validation_split_rate)


    ckpt_filename = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt_filename is not None:
        model = tf.keras.models.load_model(ckpt_filename)
    else:
        model = create_model(FLAGS.maxlen_enc, FLAGS.maxlen_dec, FLAGS.vec_size, FLAGS.num_stacks, len(data['words']), train_dataset)
        model.compile(optimizer='Adam', loss=masked_sparse_entropy, metrics=[masked_perplexity, masked_accuracy])

    model.summary()
    
    model.fit(
        train_dataset.batch(FLAGS.batch_size).prefetch(2),
        epochs=FLAGS.epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
            #   filepath=FLAGS.checkpoint_dir + "/ckpt-{epoch:06d}-{val_loss:.2f}.hdf5",
              filepath=FLAGS.checkpoint_dir + "/ckpt-{epoch:06d}.hdf5",
              save_freq=FLAGS.checkpoint_every),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(FLAGS.logdir, histogram_freq=10)
        ],
    )

    # save the final model
    print(f"Save model to {FLAGS.final_model_filepath}")
    model.save(FLAGS.final_model_filepath)

    print("Finish training")

        
        


if __name__ == '__main__':
    app.run(train)