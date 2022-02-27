import torchvision.models as models
import torch


def build_model(pretrained=True, fine_tune=False, num_classes=2):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
    else:
        print("[INFO]: Not loading pre-trained weights")

    # model = torch.hub.load("pytorch/vision:v0.11.3", "resnet152", pretrained=False)
    model = torch.hub.load(
        "pytorch/vision:v0.11.3", "efficientnet_b3", pretrained=False
    )

    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes),
        torch.nn.LogSoftmax(dim=1),
    )

    return model
