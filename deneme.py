from model_AE import AE
import torch
from torchsummary import summary

model = AE(2)
print(summary(model, (1, 56, 56)))