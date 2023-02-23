import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from .attention import MultiheadAttention
from .ffn import FFN



DROPOUT_RATE = 0.1
NUM_HEADS = 8




class Transformer(tf.keras.Model):

    def __init__(self, maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words):
        super().__init__()
        self._maxlen_enc = maxlen_enc
        self._maxlen_dec = maxlen_dec
        self._embedding_size = vec_size
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words

        self._embedding_layer = tfkl.Embedding(self._num_words, self._embedding_size)
        self._encoder = Encoder(self._maxlen_enc, self._embedding_layer, self._vec_size, self._num_stacks, self._num_words)
        self._decoder = Decoder(self._maxlen_dec, self._embedding_layer, self._vec_size, self._num_stacks, self._num_words)

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def stack_numbers(self):
        return self._num_stacks

    @property
    def vector_size(self):
        return self._vec_size

    @property
    def max_encoder_length(self):
        return self._maxlen_enc

    @property
    def max_decoder_length(self):
        return self._maxlen_dec

    @tf.function
    def call(self, inputs, training=None):
        """
        inputs: (dict)
        encoder_input: B, L
        decoder_input: B, L
        """
        encoder_output = self.encode(inputs['encoder_input'], training=training)
        decoder_output = self.decode(inputs['encoder_input'], encoder_output, inputs['decoder_input'], training=training)
        return decoder_output # B, L, NUM_WORD

    @tf.function
    def encode(self, encoder_input, training=None):
        """
        encoder_input: B, L
        """
        mask = tf.cast(tf.sign(encoder_input), tf.float32) # B,L
        return self._encoder(encoder_input, mask, training=training)

    @tf.function
    def decode(self, encoder_input, encoder_output, decoder_input, training=None):
        """
        encoder_input: (tf.Tensor) B, L
        encoder_output: (tf.Tensor) B, L, V
        decoder_input: (tf.Tensor) B, L
        """
        enc_mask = tf.cast(tf.sign(encoder_input), tf.float32) # B,L
        dec_mask = tf.cast(tf.sign(decoder_input), tf.float32) # B,L
        return self._decoder(encoder_output, enc_mask, decoder_input, dec_mask, training=training)



class Encoder(tf.keras.Model):

    def __init__(self, maxlen, embedding_layer, vec_size, num_stacks, num_words):
        super().__init__()
        self._maxlen = maxlen
        self._embedding_layer = embedding_layer
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words

        self._layer_norm = tfkl.LayerNormalization()
        self._dropout = tfkl.Dropout(DROPOUT_RATE)
        self._attention_layers = [
            MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE) for _ in range(self._num_stacks)]
        self._ffn_layers = [
            FFN(self._maxlen, self._vec_size, DROPOUT_RATE) for _ in range(self._num_stacks)]
        
    @tf.function
    def call(self, encoder_input, encoder_mask, training=None):
        """
        inputs: (dict)
        encoder_input: B, L
        decoder_mask: B, L
        """
        x = self._embedding_layer(encoder_input) # B, L, V
        # x = self._layer_norm(x)
        x *= tf.sqrt(float(self._vec_size))
        pos_emb = get_positional_encoding(self._maxlen, self._vec_size) # L, V
        x += pos_emb[None, ...] # B, L, V
        x = self._dropout(x, training=training)
        x = self._layer_norm(x)
        
        for i in range(self._num_stacks):
            x = self._attention_layers[i](
                query=x, memory=x, memory_mask=encoder_mask, training=training)
            x = self._ffn_layers[i](x, training=training)
            x *= encoder_mask[..., None]

        return x * encoder_mask[..., None]

        

class Decoder(tf.keras.Model):

    def __init__(self, maxlen, embedding_layer, vec_size, num_stacks, num_words):
        super().__init__()
        self._maxlen = maxlen
        self._embedding_layer = embedding_layer
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words

        self._layer_norm = tfkl.LayerNormalization(axis=2)
        self._dropout = tfkl.Dropout(DROPOUT_RATE)
        self._self_attention_layers = [
            MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE, apply_triangle_mask=True)
            for _ in range(self._num_stacks)]
        self._source_target_attention_layers = [
            MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE)
            for _ in range(self._num_stacks)]
        self._ffn_layers = [
            FFN(self._maxlen, self._vec_size, DROPOUT_RATE)
            for _ in range(self._num_stacks)]
        self._output_dense = tfkl.Dense(self._num_words)

    @tf.function
    def call(self, encoder_output, encoder_mask, decoder_input, decoder_mask, training=None):
        """
        encoder_output: B, L, V
        encoder_mask: B, L
        decoder_input: B, L
        decoder_mask: B, L
        """
        x = self._embedding_layer(decoder_input) # B, L, V
        # x = self._layer_norm(x)
        x *= tf.sqrt(float(self._vec_size))
        pos_emb = get_positional_encoding(self._maxlen, self._vec_size) # L, V
        x += pos_emb[None, ...] # B, L, V
        x = self._dropout(x, training=training)
        x = self._layer_norm(x)

        for i in range(self._num_stacks):
            x = self._self_attention_layers[i](
                query=x, memory=x, memory_mask=decoder_mask, training=training)
            
            # source-target attention
            x = self._source_target_attention_layers[i](
                query=x, memory=encoder_output,
                memory_mask=encoder_mask, training=training)

            # FNN
            x = self._ffn_layers[i](x, training=training)
            x *= decoder_mask[..., None]

        # apply linear layer and softmax
        x *= decoder_mask[..., None] # B,L,V
        x = self._output_dense(x)
        return x * decoder_mask[..., None]

      

def get_positional_encoding(maxlen, vec_size, epsilon=1.e-5):
    """
    PE(p, 2i) = sin(p / 10000^(2i / vec_size))
    PE(p, 2i+1) = cos(p / 10000^(2i / vec_size))

    return: L, V
    """
    evens = tf.range(vec_size, dtype=tf.float32) // 2 * 2 # 0, 0, 2, 2, 4, 4, ...
    pows = tf.pow(10000.0, evens / vec_size)
    reciprocals = tf.math.reciprocal(tf.maximum(pows, epsilon)) # V
    pos = tf.range(maxlen, dtype=tf.float32) # L
    phases = pos[:, None] @ reciprocals[None, :] # L, V
    phase_slides = tf.cast(tf.range(vec_size) % 2, tf.float32) * np.pi/2 # 0, pi/2, 0, pi/2, ...
    positional_encoding = tf.sin(phases + phase_slides[None, :]) # sin(p/10000^(0/V)), cos(p/10000^(0/V)), sin(p/10000^(2/V)), cos(p/10000^(2/V)), ...
    return positional_encoding


    
    


