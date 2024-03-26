import numpy as np

class CyraPositionalEncoding():

    """

    This Cyra model class is used to add a positioning matrix to the 
    embedding matrix.

    This method is described in the article - "Attention is all you need".

    Different words have different meanings depending on their position 
    in the text, so in addition to the meaning vector, it would be nice 
    to add a positioning vector as well.
    
    TOKEN_1 -> [num, num, num] - Positional matrix
    TOKEN_2 -> [num, num, num] - Positional matrix
    TOKEN_3 -> [num, num, num] - Positional matrix
    
    Input: Embedding Layer Output
    Output: Input + [TOKEN_1, TOKEN_2, TOKEN_3]

    """

    def __init__(self, sequence_length: int, embedding_dimension: int) -> None:

        # Creating a positional encoding matrix where 
        # each sequence token corresponds to a positioning matrix
        
        self.positional_encoding = np.zeros((sequence_length, embedding_dimension), dtype='float16')
        
        # Creating a sequence matrix filled with numbers 
        # from 0 to the sequence lenght

        self.position = np.array([i for i in range(sequence_length)], dtype='float16')

        # Adding another dimension
        # This will be required for future operations

        self.position = self.position[:, np.newaxis]
        
        # The following line of code contains an exponentially decreasing sequence
        # This sequence consists of numbers obtained by the following rule
        # 0.5 is raised to the power from the current position of this number in steps of 2

        self.descending_sequence = np.linspace(0, embedding_dimension, num=embedding_dimension // 2, endpoint=False, dtype='float16')
        self.descending_sequence = np.exp(self.descending_sequence * -(np.log(10000) / embedding_dimension), dtype='float16')

        # Next, each matrix describing the position of the token in the sequence 
        # starting from index 0 with step 2 (even) is replaced by a matrix of sine values, 
        # and all values ​​from 1 with step 2 (odd) are replaced with the cosine value
        # By multiplying the descending sequence by the current position, the frequency 
        # of the sine and cosine will be unique

        even_indices = np.arange(0, self.positional_encoding.shape[1], 2)
        odd_indices = np.arange(1, self.positional_encoding.shape[1], 2)

        self.positional_encoding[:, even_indices] = np.sin(self.position * self.descending_sequence)
        self.positional_encoding[:, odd_indices] = np.cos(self.position * self.descending_sequence)

    def __call__(self, embedding_output):

        # Since positional encoding is independent of embedding, we add 
        # it to the output of the embedding layer

        return embedding_output + self.positional_encoding

if __name__ == '__main__':

    # Test positional encoding

    cyra_pos_encoding = CyraPositionalEncoding(200, 500)

    # Print positional encoding matrix

    print('Positional Encoding')
    print(cyra_pos_encoding.positional_encoding)
    print(cyra_pos_encoding.positional_encoding.shape)

    # Print posion matrix

    print('Position')
    print(cyra_pos_encoding.position)
    print(cyra_pos_encoding.position.shape)

    #  Exponentially decreasing sequence

    print('Descending Sequence')
    print(cyra_pos_encoding.descending_sequence)
    print(cyra_pos_encoding.descending_sequence.shape)

    #  Positional encoding

    print('Positional Encoding')
    print(cyra_pos_encoding.positional_encoding)
    print(cyra_pos_encoding.positional_encoding.shape)