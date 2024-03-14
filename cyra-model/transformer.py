from keras.layers import Dropout, LayerNormalization, Input
from keras.layers import Attention
from keras.regularizers import l2
from keras.models import Model

class CyraTransformer:

    def __init__(self, vocab_size, embedding_dim, num_layers, num_neurons, dropout_rate=0.1, reg_lambda=0.01) -> None:
        
        self.inputs = Input(shape=(None,))
        self.outputs = self.inputs
        
        for _ in range(num_layers):

            self.attention = Attention(num_neurons, kernel_regularizer=l2(reg_lambda))([self.outputs, self.outputs])
            self.outputs = self.layernorm(self.dropout(self.attention) + self.outputs)

        self.model = Model(self.inputs, self.outputs)

    def __call__(self, inputs) -> None:
        
        return self.model(inputs)