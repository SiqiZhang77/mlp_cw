import unittest
import torch
from pytorch_mlp_framework.model_architectures import(
    ConvolutionalProcessingBlock_BN,
    ConvolutionalDimReductionBlock_BN,
    ConvolutionalProcessingBlock_BN_ResCon,
)

class TestBlocks(unittest.TestCase):
    def setUp(self):
        # set parameter
        self.input_shape = (4, 3, 32, 32)  # batch_size, channels, height, width
        self.num_filters = 16
        self.kernel_size = 3
        self.padding = 1
        self.bias = False
        self.dilation = 1
        self.reduction_factor = 2

    def test_convolutional_processing_block_bn(self):
        # test ConvolutionalProcessingBlock_BN
        block = ConvolutionalProcessingBlock_BN(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
        )
        x = torch.randn(self.input_shape)  # 
        output = block.forward(x)  # 
        self.assertEqual(output.shape[1], self.num_filters)  # test output

    def test_convolutional_dim_reduction_block_bn(self):
        # test ConvolutionalDimReductionBlock_BN
        block = ConvolutionalDimReductionBlock_BN(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
            reduction_factor=self.reduction_factor,
        )
        x = torch.randn(self.input_shape)  #generate random tensor
        output = block.forward(x)  
        self.assertEqual(output.shape[2], self.input_shape[2] // self.reduction_factor)  # test height
        self.assertEqual(output.shape[3], self.input_shape[3] // self.reduction_factor)  

    def test_convolutional_processing_block_bn_rescon(self):
        # Test ConvolutionalProcessingBlock_BN_ResCon
        block = ConvolutionalProcessingBlock_BN_ResCon(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
        )
        x = torch.randn(self.input_shape)  # Generate random tensor
        output = block.forward(x)  
        self.assertEqual(output.shape, x.shape)  # Test shape consistency


if __name__ == '__main__':
    unittest.main()
