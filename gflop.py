import torch
import torch.nn as nn
from thop import profile
from torchvision import models
from mgca_module_decomp_concatenate import MGCA
# Load ResNet18 model
#model = MGCA().cuda()
# Example: Load a model (e.g., ResNet-50)
model = models.resnet50().cuda()

# Example input size for the model (batch_size=1, 3 channels, 224x224 image)
input_data1 = torch.randn(1, 3, 224, 224)
input_data2 = torch.randn(1, 3, 224, 224)
input_data1 = input_data1.cuda()
input_data2 = input_data2.cuda()
# Use thop to profile the model (calculates both FLOPs and number of parameters)
flops, params = profile(model, inputs=(input_data1, ))

# Convert FLOPs to GFLOPs
gflops = flops / 1e9

print(f"Total FLOPs: {flops}")
print(f"Total GFLOPs: {gflops:.2f} GFLOPs")
print(f"Total Parameters: {params}")
