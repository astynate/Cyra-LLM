from tensorflow.keras.layers import Dropout, LayerNormalization, Layer
from tensorflow.keras.layers import Dense, MultiHeadAttention
from tensorflow.keras.models import Sequential
# from cyra_model.multihead_attention import MultiHeadAttention
import tensorflow as tf

class TransformerBlock():

    def __init__(self, embedding_dim, num_heads, feed_forward_dim):
        
        self.self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)

        self.linear_input = Dense(feed_forward_dim, activation="relu")
        self.linear_output = Dense(embedding_dim)

    def __call__(self, inputs, attention_mask):

        # Here query, key and value have the same meaning, 
        # which is why this mechanism is called self-attention

        attention_output = self.self_attention(inputs, inputs, inputs, attention_mask=attention_mask)
        normalized_attention_output = LayerNormalization()(inputs + attention_output)

        feed_forward_output = Sequential([
            self.linear_input,
            self.linear_output
        ])(normalized_attention_output)

        feed_forward_output = Dropout(0.1)(feed_forward_output)

        return LayerNormalization()(normalized_attention_output + feed_forward_output)