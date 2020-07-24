from workspace_utils import active_session
from torchvision import transforms
import time
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}


data_dir = 'flowers'
traindir = data_dir + '/train'
validdir = data_dir + '/valid'
testddir = data_dir + '/valid'


# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root=validdir, transform=image_transforms['test']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=2, shuffle=True),
    'val': DataLoader(data['valid'], batch_size=2, shuffle=True),
    'test': DataLoader(data['test'], batch_size=2, shuffle=True)
}


model = models.vgg16(pretrained=True)


# Custom Classifier
classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

# Freeze the parameters of densenet so that losses doen't back propagate
for param in model.features.parameters():
    param.require_grad = False

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Move to gpu
model = model.to('cuda')

n_epochs = 2
loss_train = 0
steps = 0

with active_session():
    start = time.time()
    for epoch in range(n_epochs):
        for data, targets in dataloaders['train']:
            steps += 1
            # 설명
            optimizer.zero_grad()
            # Generate predictions
            out = model(data.to('cuda'))
            # Calculate loss
            loss = criterion(out, targets.to('cuda'))
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            #
            loss_train = loss.item()

            if steps%2 == 0:
                valid_loss = 0
                acc = 0
                model.eval()
                with torch.no_grad():
                    for data, targets in dataloaders['val']:
                        out = model(data.to('cuda'))
                        valid_loss += criterion(out, targets.to('cuda'))
                        # ??
                        ps = torch.exp(out)
                        #??
                        top_prob, top_class = ps.topk(1,dim=1)
                        equals = top_class == targets.to('cuda').view(*top_class.shape)
                        acc += torch.mean(equals.type(torch.FloatTensor))

                    print(f"epoch: {epoch+1}/{n_epochs}...")
                    print(f"train loss: {loss_train/2:.3f}")
                    print(f"validation loss: {valid_loss/len(dataloaders['val']):.3f}")
                    print(f"validation acc: {acc/len(dataloaders['val']):.3f}")

                    loss_train = 0
                    model.train()

    end = time.time()
    print((end-start)/60, 'minutes')
    torch.save(model.state_dict(), 'checkpoint.pth')
