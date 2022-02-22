import torch

model = torch.hub.load("pytorch/vision:v0.8.2", "resnet152", pretrained=False)
print(model)
