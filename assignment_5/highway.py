#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.highway_proj = nn.Linear(e_word, e_word)
        self.highway_gate = nn.Linear(e_word, e_word)

    def forward(self, x_conv_out):
        x_proj = torch.relu(self.highway_proj(x_conv_out))
        x_gate = torch.sigmoid(self.highway_gate(x_conv_out))
        return x_gate * x_proj + (1 - x_gate) * x_conv_out
### END YOUR CODE 
