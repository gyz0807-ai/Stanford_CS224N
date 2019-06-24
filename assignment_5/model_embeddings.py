#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

import torch
from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        e_char = 50
        kernel_size = 5
        dropout_prob = 0.3
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), e_char, padding_idx=pad_token_idx)
        self.cnn = CNN(e_char=e_char, e_word=embed_size, kernel_size=kernel_size)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(dropout_prob)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        word_embed_out = []
        embed_out = self.embeddings(input)

        for s_embedded in embed_out:
            s_embedded_reshaped = s_embedded.permute(0, 2, 1)
            cnn_out = self.cnn(s_embedded_reshaped)
            highway_out = self.highway(cnn_out)
            dropout_out = self.dropout(highway_out)
            word_embed_out.append(torch.unsqueeze(dropout_out, 0))

        word_embed_out = torch.cat(word_embed_out)
        return word_embed_out
        ### END YOUR CODE

