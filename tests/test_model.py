import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

class TestModelSpecs(unittest.TestCase):
    def setUp(self):
        self.model = Net()
    
    def test_parameter_count(self):
        """Test if model has less than 20k parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 20000, "Model has more than 20k parameters")
    
    def test_batch_norm_presence(self):
        """Test if model uses batch normalization"""
        has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in self.model.modules())
        self.assertTrue(has_batch_norm, "Model does not use batch normalization")
    
    def test_dropout_presence(self):
        """Test if model uses dropout"""
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        self.assertTrue(has_dropout, "Model does not use dropout")
    
    def test_gap_presence(self):
        """Test if model uses Global Average Pooling"""
        has_gap = any(isinstance(m, nn.AvgPool2d) for m in self.model.modules())
        self.assertTrue(has_gap, "Model does not use Global Average Pooling")

if __name__ == '__main__':
    unittest.main() 