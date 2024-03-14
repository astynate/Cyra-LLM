import tensorflow as tf
from tokenizer import CyraTokenizer
from transformer import CyraTransformer
from keras.layers import Input, Dense, Embedding
from keras.optimizers import Adam
from keras.models import Model
from positional_encoding import PositionalEncoding

class Cyra:

    def __init__(self, transformer_block_counter, embedding_dim, num_layers, num_neurons) -> None:

        # External dependencies

        self.tokenizer = CyraTokenizer()

        # Neural network layers

        self.inputs = Input(shape=(None,))

        self.embedding = Embedding(self.tokenizer.get_dimension(), embedding_dim)(self.inputs)
        self.pos_encoding = PositionalEncoding(embedding_dim, embedding_dim)(self.embedding)

        self.transformer_block = CyraTransformer(embedding_dim, num_layers, num_neurons)(self.pos_encoding)

        for _ in range(transformer_block_counter - 1):

            self.transformer_block = CyraTransformer(embedding_dim, num_layers, num_neurons)(self.transformer_block)

        self.outputs = Dense(self.tokenizer.get_dimension())
        self.model = Model(inputs=self.inputs, outputs=self.outputs)