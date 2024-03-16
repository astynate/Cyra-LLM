from keras.layers import Dropout, LayerNormalization, Layer
from keras.layers import MultiHeadAttention, Dense
from keras.models import Sequential

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        print('-------------------------------------------------')
        print(f'attn_output: {attn_output.shape}')
        print(attn_output)

        out1 = self.layernorm1(inputs + attn_output)
        print('-------------------------------------------------')
        print(f'out1: {out1.shape}')
        print(out1)

        ffn_output = self.ffn(out1)
        print('-------------------------------------------------')
        print(f'ffn_output: {ffn_output.shape}')
        print(ffn_output)

        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config