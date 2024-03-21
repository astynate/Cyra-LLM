from cyra_model.tokenizer import CyraTokenizer
from cyra_model.transformer import TransformerBlock
from cyra_model.positional_encoding import CyraPositionalEncoding
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LayerNormalization, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

mixed_precision.set_global_policy('mixed_float16')

class GradientClippingOptimizer(tf.keras.optimizers.Adam):
    def get_gradients(self, loss, params):

        gradients = super().get_gradients(loss, params)

        clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        return clipped_gradients

custom_objects = {
    'PositionEncoding': CyraPositionalEncoding,
    'TransformerBlock': TransformerBlock,
    'GradientClippingOptimizer': GradientClippingOptimizer
}

def create_attention_mask(input):
    seq = tf.cast(tf.math.equal(input, 0), tf.float16)
    return seq[:, tf.newaxis, tf.newaxis, :]

class Cyra:
    def __init__(self, tokenizer, transformer_block_counter, embedding_dim, num_heads, feed_forward_dim, **kwargs) -> None:
        self.tokenizer = tokenizer

        self.inputs = Input(shape=(self.tokenizer.sequence_length,))
        self.embedding = Embedding(input_dim=self.tokenizer.get_dimension(), output_dim=embedding_dim)(self.inputs)

        self.pos_encoding = CyraPositionalEncoding(self.tokenizer.sequence_length, embedding_dim)(self.embedding)
        self.pos_encoding = Dropout(0.1)(self.pos_encoding)

        self.transformer_block = self.pos_encoding
        self.attention_mask = create_attention_mask(self.inputs)

        for _ in range(transformer_block_counter):
            self.transformer_block = TransformerBlock(
                embedding_dim, 
                num_heads, 
                feed_forward_dim
            )(
                self.transformer_block,
                attention_mask=self.attention_mask
            )

        self.transformer_block = Flatten()(self.transformer_block)
        self.transformer_block = LayerNormalization()(self.transformer_block)

        self.outputs = Dense(self.tokenizer.get_dimension(), activation='softmax', kernel_initializer='glorot_uniform')(self.transformer_block)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

        if (kwargs.get('path') and os.path.isfile(kwargs.get('path'))):
            self.model.load_weights(kwargs.get('path'))
            print(f'Cyra model was loaded, count params: {self.model.count_params()}')

        else:
            print(f'Cyra model was created, count params: {self.model.count_params()}')

        print(f'Input shape: {self.inputs.shape}')
        print(f'Ouput shape: {self.outputs.shape}')

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        self.model.compile(optimizer=GradientClippingOptimizer(learning_rate=lr_schedule), 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['sparse_categorical_accuracy'])

    def __call__(self, text: str) -> str:

        tokens = self.tokenizer.get_sequence(text)
        tokens = np.array(tokens).reshape(1, 50)
        
        predicted_label = self.model.predict(np.array(tokens))
        predicted_word = self.tokenizer.get_text([[np.argmax(predicted_label[0])]][0])

        return predicted_word

if __name__ == '__main__':
    
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 1, 1024, 16, 1024)