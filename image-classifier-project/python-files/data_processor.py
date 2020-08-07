import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


def transform_image():
    """Returns transform object of train/valid/test datasets."""
    image_transforms = {
        'train':
        transforms.Compose([
            # use data augmentation on training sets
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # no augmentation on validation and test sets
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return image_transforms


def get_dataloaders(data_dir):
    """Return data and dataloader based on data directory

    Args: string value that refers file directory

    Returns:
        data: train/valid/test datasets dictionary
        dataloaders: train/valid/test dataloader dictionary
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Image transformations
    image_transforms = transform_image()
    data = {
        'train':
        datasets.ImageFolder(root=train_dir,
                             transform=image_transforms['train']),
        'valid':
        datasets.ImageFolder(root=valid_dir,
                             transform=image_transforms['valid']),
        'test':
        datasets.ImageFolder(root=test_dir,
                             transform=image_transforms['test']),
    }
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=32, shuffle=True),
        'val': DataLoader(data['valid'], batch_size=32),
        'test': DataLoader(data['test'], batch_size=32)
    }
    return data, dataloaders


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,

    returns an Numpy array
    """
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
