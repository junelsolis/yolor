import wandb
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

data_dir = "resnet_dataset"


def load_split_train_test(datadir, valid_size=0.25):
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=64
    )
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=64
    )

    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, 0.25)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("pytorch/vision:v0.11.3", "resnet152", pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 10),
    torch.nn.LogSoftmax(dim=1),
)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

model.to(device)

epochs = 60
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

wandb.login()
wandb.init(project="ihc-classifier", entity="junelsolis")
wandb.config = {
    "learning_rate": 0.003,
    "optimizer": "Adam",
    "epochs": epochs,
    "batch_size": 64,
}

# wandb.log({'lo'})

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))
            print(
                f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(testloader):.3f}"
            )
            running_loss = 0
            wandb.log(
                {
                    "train_loss": running_loss / len(trainloader),
                    "test_loss": test_loss / len(testloader),
                    "test_accuracy": accuracy / len(testloader),
                    "epoch": epoch + 1,
                }
            )
            model.train()

torch.save(model, "ihc-classifier-01.pth")
