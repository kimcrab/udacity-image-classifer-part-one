from torchvision import models
import torchvision
import torch
import json
import torch.optim as optim


def get_device_mode(gpu):    
    if gpu is True:
        device = "cuda"
        print("set to GPU mode")
    else:
        device = "cpu"
        print("set to CPU mode")
    
    return device

def get_model_arch(arch):
    if arch == "vgg16":
        print("model set to vgg16")
        model = models.vgg16(pretrained=True)
    else:
        print("currently only support vgg16")
        print("model set to vgg16")
        model = models.vgg16(pretrained=True)
    
    return model

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def map_categories(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

