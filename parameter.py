import torch
import torch.nn as nn
from torchvision import models
from mgca_module_decomp_concatenate import MGCA
# Load ResNet18 model
model = MGCA()

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Calculate the size in MB (assuming float32, 4 bytes per parameter)
model_size = total_params * 4 / (1024 ** 2)

print(f"ResNet-18 has {total_params} parameters.")
print(f"Model size: {model_size:.2f} MB")
