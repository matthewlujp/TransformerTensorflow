import pathlib
from absl import app, flags
import numpy as np
import tensorflow as tf

from utils import load_data, create_dataset, create_model


FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 100, "number of epochs", short_name='e')
flags.DEFINE_integer("batch_size", 64, "batch size", short_name='b')
flags.DEFINE_string("savedir", None, "path to save training results", short_name='s')
flags.DEFINE_string("checkpoint_file", None, "path to a checkpoint file to laod", short_name='c')
flags.DEFINE_string("data_filepath", None, "path to data", short_name='d')
flags.DEFINE_float("validation_split_rate", 0.1, "ration of validation data in total", short_name='srate')
flags.DEFINE_integer("maxlen_enc", 50, "max words in an input prompt")
flags.DEFINE_integer("maxlen_dec", 50, "max words in a model output")
flags.DEFINE_integer("vec_size", 512, "internal vector size")
flags.DEFINE_integer("num_stacks", 6, "number of modules to stack in encoder and decoder")
flags.DEFINE_bool('debug', False, "run in debug mode")





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
    sentence_lengths = tf.reduce_sum(mask, -1) # B
    acc = tf.cast(tf.equal(y_true, tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)), dtype=tf.float32) # B, L
    masked_acc = tf.reduce_sum(acc * mask, axis=-1) / sentence_lengths # B, L
    return tf.reduce_mean(masked_acc)
    

    



class UpdateLearningRate(tf.keras.callbacks.Callback):
    warmup_steps = 4000.0
    # warmup_steps = 8000.0
    max_lr = 0.0001

    def on_batch_begin(self, batch, logs=None):
        step = np.maximum(self.model.optimizer.iterations.numpy(), 0.5)
        # lr = tf.pow(float(self.model.vector_size), -0.5) \
        #     * tf.minimum(tf.pow(float(step), -0.5), step * tf.pow(self.warmup_steps, -1.5))
        lr = self.max_lr * min(step ** -0.5, step * self.warmup_steps ** -1.5) / self.warmup_steps ** -0.5
        self.model.optimizer.lr.assign(lr)




def train(_):
    savedir = pathlib.Path(FLAGS.savedir)
    logdir = savedir / 'logdir'
    ckptdir = savedir / 'ckpt'
    tmpdir = savedir / 'tmp'

    if not savedir.exists():
        savedir.mkdir(parents=True)
        logdir.mkdir()
        ckptdir.mkdir()
        

    data = load_data(FLAGS.data_filepath)
    train_dataset, val_dataset = create_dataset(data, FLAGS.validation_split_rate)

    if FLAGS.debug:
        # with tf.device('CPU'):
        model = create_model(FLAGS.maxlen_enc, FLAGS.maxlen_dec, 16, 2, len(data['words']), train_dataset)
        train_dataset = train_dataset.take(80).batch(10).prefetch(3)
        val_dataset = val_dataset.take(40).batch(10)
    else:
        model = create_model(FLAGS.maxlen_enc, FLAGS.maxlen_dec, FLAGS.vec_size, FLAGS.num_stacks, len(data['words']), train_dataset)
        train_dataset = train_dataset.batch(FLAGS.batch_size).prefetch(3)
        val_dataset = val_dataset.batch(50).prefetch(3)
        

    if FLAGS.checkpoint_file is not None:
        ckpt_filepath = pathlib.Path(FLAGS.checkpoint_file)
        print(f"Loading checkpoint from {str(ckpt_filepath)}.")
        model.load_weights(ckpt_filepath)

    model.compile(
        optimizer='Adam', loss=masked_sparse_entropy,
        metrics=[masked_perplexity, masked_accuracy])
    model.summary()
    model.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.BackupAndRestore(backup_dir=tmpdir, delete_checkpoint=True),
            tf.keras.callbacks.ModelCheckpoint(
              filepath=str(ckptdir) + "/checkpoint-epoch={epoch:02d}-val_loss={val_loss:.02f}.hdf5",
              save_freq='epoch', save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, start_from_epoch=50, verbose=1),
            tf.keras.callbacks.TensorBoard(logdir, histogram_freq=10),
            UpdateLearningRate(),
        ],
    )

    # save the final weights
    final_weight_filepath = savedir / f"final_weights.h5"
    print(f"Save weights to {final_weight_filepath}")
    model.save_weights(final_weight_filepath, save_format='h5')
    # save model
    final_model_filepath = savedir / f"final_model"
    print(f"Save model to {final_model_filepath}")
    model.save(
        final_model_filepath,
        signatures={
            'serving_default': model.call.get_concrete_function({
                'encoder_input': tf.TensorSpec([None, model.max_encoder_length], dtype=tf.int32, name='encoder_input'),
                'decoder_input': tf.TensorSpec([None, model.max_decoder_length], dtype=tf.int32, name='decoder_input')}),
            'encode': model.encode.get_concrete_function(tf.TensorSpec([None, model.max_encoder_length], dtype=tf.int32, name='encoder_input')),
            'decode': model.decode.get_concrete_function(
                encoder_input=tf.TensorSpec([None, model.max_encoder_length], dtype=tf.int32, name='encoder_input'),
                encoder_outputs=tf.TensorSpec([None, model.max_encoder_length, model.vector_size], dtype=tf.float32, name='encoder_outputs'),
                decoder_input=tf.TensorSpec([None, model.max_decoder_length], dtype=tf.int32, name='decoder_input')),
        })
    print("Finish training")

        
        


if __name__ == '__main__':
    app.run(train)