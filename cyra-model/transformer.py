from keras.layers import Dropout, LayerNormalization, Input
from keras.layers import Attention
from keras.models import Model

class CyraTransformer:

    def __init__(self, num_layers, num_neurons, dropout_rate=0.1) -> None:
        
        self.inputs = Input(shape=(None,))
        self.outputs = self.inputs
        
        for _ in range(num_layers):
            
            self.attention = Attention()([self.outputs, self.outputs])
            self.dropout = Dropout(dropout_rate)
            self.layernorm = LayerNormalization(epsilon=1e-6)
            self.outputs = self.layernorm(self.dropout(self.attention) + self.outputs)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def __call__(self, inputs) -> None:

        return self.model(inputs)
    
# if __name__ == '__main__':
#     ct = CyraTransformer(1, 1)()