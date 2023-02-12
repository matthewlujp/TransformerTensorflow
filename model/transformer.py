import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from .attention import MultiheadAttention
from .ffn import FFN



DROPOUT_RATE = 0.1
NUM_HEADS = 8




class Transformer(tf.keras.Model):

    def __init__(
            self, maxlen_enc, maxlen_dec, vec_size, num_stacks, num_words):
        self._maxlen_enc = maxlen_enc
        self._maxlen_dec = maxlen_dec
        self._embedding_size = embedding_size
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words

        self._embedding_layer = tfkl.Embedding(self._num_words, self._embedding_size)
        self._encoder = Encoder(self._maxlen_enc, self._embedding_layer, self._vec_size, self._num_stacks, self._num_words)
        self._decoder = Decoder(self._maxlen_dec, self._embedding_layer, self._vec_size, self._num_stacks, self._num_words)


    def call(self, inputs):
        """
        inputs: (dict) encoder_input, decoder_input
        """
        e_mask = tf.cast(tf.sign(inputs['encoder_input']), tf.float32) # B,L
        encoder_outputs = self._encoder({
            'encoder_input': inputs['encoder_input'],
            'encoder_mask': e_mask,
        })
        d_mask = tf.cast(tf.sign(inputs['decoder_input']), tf.float32) # B,L
        decoder_output = self._decoder({
            'encoder_outputs': encoder_outputs,
            'encoder_mask': e_mask,
            'decoder_input': inputs['decoder_input'],
            'decoder_mask': d_mask,
        })
        return decoder_output # B, L, NUM_WORD
        



class Encoder(tf.keras.Model):

    def __init__(self, maxlen, embedding_layer, vec_size, num_stacks, num_words):
        self._maxlen = maxlen
        self._embedding_layer
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words
        
    def call(self, inputs):
        """
        inputs: (dict) encoder_input, encoder_mask
        """
        e_i = inputs['encoder_input']
        e_mask = inputs['encoder_mask']

        x = self._embedding_layer(e_i) # B, L, V
        x = tfkl.LayerNormalization()(x)
        x *= tf.squrt(self._vec_size)
        pos_emb = get_positional_encoding(self._maxlen, self._vec_size) # L, V
        x += pos_emb[None, ...] # B, L, V
        x = tfkl.Dropout(DROPOUT_RATE)(x)
        
        encoder_outputs = []
        for i in range(self._num_stacks):
            x = MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE)(
                query=x, key=x, value=x, query_mask=e_mask, value_mask=e_mask)
            x = FFN(self._maxlen, self._vec_size, DROPOUT_RATE)(x)
            x *= e_mask[..., None]
            encoder_outputs.append(x)

        return encoder_outputs

        

class Decoder(tf.keras.Model):

    def __init__(self, maxlen, embedding_layer, vec_size, num_stacks, num_words):
        self._maxlen = maxlen
        self._embedding_layer
        self._vec_size = vec_size
        self._num_stacks = num_stacks
        self._num_words = num_words

    def call(self, inputs):
        """
        inputs: (dict) encoder_outputs, decoder_input, encoder_mask, decoder_mask
        """
        d_i = inputs['decoder_input'] # B, L
        d_mask = inputs['decoder_mask'] # B, L
        encoder_outputs = inputs['encoder_outputs']
        e_mask = inputs['encoder_mask']

        x = self._embedding_layer(d_i) # B, L, V
        x = tfkl.LayerNormalization()(x)
        x *= tf.sqrt(self._vec_size)
        pos_emb = get_positional_encoding(self._maxlen, self._vec_size) # L, V
        x += pos_emb[None, ...] # B, L, V
        x = tfkl.Dropout(DROPOUT_RATE)(x)

        for i in range(self._num_stacks):
            # self attention
            x = MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE)(
                query=x, key=x, value=x, query_mask=d_mask, value_mask=d_mask, triangle_mask=True)
            
            # source-target attention
            x = MultiheadAttention(self._vec_size, NUM_HEADS, DROPOUT_RATE, apply_triangle_mask=True)(
                query=x, key=encoder_outputs[i], value=encoder_outputs[i], query_mask=d_mask)

            # FNN
            x = FNN(self._maxlen, self._vec_size, DROPOUT_RATE)(x)
            x *= d_mask[..., None]

        # apply linear layer and softmax
        x *= d_mask[..., None] # B,L,V
        x = tfkl.Dense(self._num_words)(x)
        # decoder_output = tfkl.Softmax()(x)
        decoder_output = x # output logit
        return decoder_output

            





    
    
            




        



def get_positional_encoding(maxlen, vec_size, epsilon=1.e-5):
    """
    PE(p, 2i) = sin(p / 10000^(2i / vec_size))
    PE(p, 2i+1) = cos(p / 10000^(2i / vec_size))
    """
    evens = tf.range(vec_size, dtype=tf.float32) // 2 * 2 # 0, 0, 2, 2, 4, 4, ...
    pows = tf.pow(10000, evens / vec_size)
    reciprocals = tf.math.reciprocals(tf.maximum(pows, epsilon)) # V
    pos = tf.range(maximum, dtype=tf.float32) # L
    phases = pos[:, None] @ reciprocals[None, :] # L, V
    phase_slides = tf.cast(tf.range(vec_size) % 2, tf.float32) * np.pi # 0, pi, 0, pi, ...
    positional_encoding = tf.sin(phases + phase_slides[None, :]) # sin(p/10000^(0/V)), cos(p/10000^(0/V)), sin(p/10000^(2/V)), cos(p/10000^(2/V)), ...
    return positional_encoding


    
    


