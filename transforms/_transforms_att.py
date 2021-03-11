#!/usr/bin/env python3

import torch
import random
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "AttMask",
    "AttToTensor",        
]

class AttToTensor(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return torch.from_numpy(data)

    def __repr__(self):
        return self.__class__.__name__
        
        
class AttMask(object):
    def __init__(self, mask_ratio=0.05, replace_with_zero=True):
        self.mask_ratio = mask_ratio
        self.replace_with_zero = replace_with_zero                

    def __call__(self, att):
        copy = att.copy()
        att_dim = att.shape[0]
        mask_num = int(self.mask_ratio*att_dim)
          
        idx = np.arange(att_dim)
        random.shuffle (idx)
        
        if (self.replace_with_zero): att[idx[:mask_num]] = 0
        else: att[idx[:mask_num]] = copy.mean()
    
        return att

    def __repr__(self):
        return self.__class__.__name__
   