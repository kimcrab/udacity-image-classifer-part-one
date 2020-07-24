# Imports here
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
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from PIL import Image
import os, random
from torch.autograd import Variable
import torchvision


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
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

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root=test_dir, transform=image_transforms['test']),
}


# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=32, shuffle=True),
    'val': DataLoader(data['valid'], batch_size=32, shuffle=True),
    'test': DataLoader(data['test'], batch_size=32, shuffle=True)
}


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    # TODO: Build and train your network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.vgg16(pretrained=True)

# Freeze the parameters of densenet so that losses doen't back propagate
for param in model.parameters():
    param.requires_grad = False


# Custom Classifier
classifier = nn.Sequential(
    nn.Linear(25088, 4096, bias=True),
    nn.ReLU(),
    nn.Dropout(),
#     nn.Linear(4096, 4096, bias=True),
#     nn.ReLU(),
#     nn.Dropout(),
    nn.Linear(4096, 102, bias=True),
    nn.LogSoftmax(dim=1)
)


model.classifier = classifier

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



# Move to gpu
model = model.to(device)



with active_session():
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 10

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
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
                    for inputs, labels in dataloaders['val']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(dataloaders['train']))
                test_losses.append(test_loss/len(dataloaders['val']))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['val']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['val']):.3f}")
                running_loss = 0
                model.train()



plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

# TODO: Do validation on the test set
test_accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {test_accuracy/len(dataloaders['test']):.3f}")

# TODO: Save the checkpoint
model.class_to_idx = data['train'].class_to_idx

checkpoint = {'input_size': 25088,
              'hidden_layers':[4096],
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.001,
              'batch_size': 32,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


nn_filename = 'checkpoint.pth'

model, optimizer = load_checkpoint(nn_filename)

saved_model = print(model)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img_size = 256
    crop_size = 224

    im = Image.open(image)
    im = im.resize((img_size,img_size))

    left = (img_size-crop_size)*0.5
    right = left + crop_size
    upper = (img_size-crop_size)*0.5
    lower = upper + crop_size

    im = im.crop((left, upper, right, lower))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    image = process_image(image_path)
    # ??
    image = torch.from_numpy(np.array([image])).float()

    image = Variable(image)
    # ??

    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, labels = torch.topk(ps, topk)
        # ??
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}

        classes = []

        for label in labels.numpy()[0]:
            classes.append(class_to_idx_rev[label])

        return probs.numpy()[0], classes


img = random.choice(os.listdir('./flowers/test/56/'))
img_path = './flowers/test/56/' + img
with  Image.open(img_path) as image:
    plt.imshow(image)

prob, classes = predict(img_path, model)
print(prob)
print(classes)
print([cat_to_name[x] for x in classes])



# TODO: Display an image along with the top 5 classes
probs, classes = predict(img_path, model)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)


ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(Image.open(img_path))

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])

y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

plt.show()
