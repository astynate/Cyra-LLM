from cyra_model.tokenizer import CyraTokenizer
from cyra_model.transformer import TransformerBlock
from cyra_model.positional_encoding import CyraPositionalEncoding
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LayerNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np

mixed_precision.set_global_policy('mixed_float16')

custom_objects = {
    'PositionEncoding': CyraPositionalEncoding,
    'TransformerBlock': TransformerBlock
}

class Cyra:

    def __init__(self, tokenizer, transformer_block_counter, embedding_dim, num_heads, feed_forward_dim) -> None:

        self.tokenizer = tokenizer
        self.inputs = Input(shape=(self.tokenizer.sequence_length,))

        self.embedding = Embedding(input_dim=self.tokenizer.get_dimension(), output_dim=embedding_dim)(self.inputs)
        self.pos_encoding = CyraPositionalEncoding(self.tokenizer.sequence_length, embedding_dim)(self.embedding)

        self.transformer_block = self.pos_encoding
        self.attention_mask = np.ones_like(self.tokenizer.sequence_length)

        for _ in range(transformer_block_counter):
            self.transformer_block = TransformerBlock(
                embedding_dim, 
                num_heads, 
                feed_forward_dim
            )(
                self.transformer_block, 
                attention_mask = np.ones((self.tokenizer.sequence_length,), dtype='float16').reshape((1, 1, self.tokenizer.sequence_length))
            )

        self.transformer_block = Flatten()(self.transformer_block)
        self.transformer_block = LayerNormalization()(self.transformer_block)

        self.outputs = Dense(self.tokenizer.get_dimension(), activation='softmax')(self.transformer_block)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.002), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        print(f'Input shape: {self.inputs.shape}')
        print(f'Ouput shape: {self.outputs.shape}')
        print(f'Cyra model was created, count params: {self.model.count_params()}')

    def __call__(self, text: str) -> str:

        tokens = self.tokenizer.get_sequence(text)
        tokens = np.array(tokens).reshape(1, 50)
        
        predicted_label = self.model.predict(np.array(tokens))
        predicted_word = self.tokenizer.get_text([[np.argmax(predicted_label[0])]][0])

        return predicted_word

if __name__ == '__main__':
    
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 12, 1024, 16, 1024)