import numpy as np
import tensorflow as tf
from keras.layers import Embedding

vocab_size = 10000
embedding_dim = 512

inputs = np.ones(shape=(1, 50), dtype="int32")

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

outputs = embedding_layer(inputs)

print(embedding_layer.count_params())
print(outputs)