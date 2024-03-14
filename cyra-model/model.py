import tensorflow as tf
from tokenizer import CyraTokenizer
from transformer import CyraTransformer
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from positional_encoding import PositionalEncoding

class Cyra:

    def __init__(self, tokenizer, transformer_block_counter, embedding_dim, num_layers, num_neurons) -> None:

        self.tokenizer = tokenizer
        self.inputs = Input(shape=(None,))

        self.embedding = Embedding(self.tokenizer.get_dimension(), embedding_dim)(self.inputs)
        self.pos_encoding = PositionalEncoding(embedding_dim, embedding_dim)(self.embedding)

        self.transformer_block = CyraTransformer(num_layers, num_neurons)(self.pos_encoding)
        self.transformer_block = BatchNormalization()(self.transformer_block)
        self.transformer_block = Dropout(0.1)(self.transformer_block)

        for _ in range(transformer_block_counter - 1):
            self.transformer_block = CyraTransformer(num_layers, num_neurons)(self.transformer_block)
            self.transformer_block = BatchNormalization()(self.transformer_block)
            self.transformer_block = Dropout(0.1)(self.transformer_block)

        self.outputs = Dense(self.tokenizer.get_dimension(), activation='softmax')(self.transformer_block)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        print(f'Cyra model was created, count params: {self.model.count_params()}')

if __name__ == '__main__':
    
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 2, 512, 4, 1024)