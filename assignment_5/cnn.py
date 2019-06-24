#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, e_char, e_word, kernel_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(e_char, e_word, kernel_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x_reshaped):
        conv_out = torch.relu(self.conv(x_reshaped))
        max_pool_out = self.max_pool(conv_out)
        return torch.squeeze(max_pool_out, -1)
### END YOUR CODE
