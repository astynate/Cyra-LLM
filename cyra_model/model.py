from tokenizer import CyraTokenizer
from transformer import TransformerBlock
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from positional_encoding import PositionalEncoding
from keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

custom_objects = {
    'PositionEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
}

class Cyra:
    def __init__(self, tokenizer, transformer_block_counter, embedding_dim, num_heads, feed_forward_dim) -> None:
        self.tokenizer = tokenizer
        self.inputs = Input(shape=(None,))

        self.embedding = Embedding(input_dim=self.tokenizer.get_dimension(), output_dim=embedding_dim)(self.inputs)
        self.pos_encoding = PositionalEncoding(embedding_dim, embedding_dim)(self.embedding)

        self.transformer_block = TransformerBlock(embedding_dim, num_heads, feed_forward_dim)(self.pos_encoding)
        self.transformer_block = BatchNormalization()(self.transformer_block)
        self.transformer_block = Dropout(0.1)(self.transformer_block)

        for _ in range(transformer_block_counter - 1):
            self.transformer_block = TransformerBlock(embedding_dim, num_heads, feed_forward_dim)(self.transformer_block)
            self.transformer_block = BatchNormalization()(self.transformer_block)
            self.transformer_block = Dropout(0.1)(self.transformer_block)

        self.outputs = Dense(self.tokenizer.get_dimension(), activation='softmax')(self.transformer_block)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        print(f'Cyra model was created, count params: {self.model.count_params()}')

if __name__ == '__main__':
    
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 12, 1024, 16, 1024)