import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

from arguments import get_training_args
from data_processor import get_dataloaders
from utils import get_device_mode, get_model_arch
from workspace_utils import active_session

# Set command line arguments for training.
parser = get_training_args()
args = parser.parse_args()
# Load the data and dataloaders.
data, dataloaders = get_dataloaders(args.data_dir)
# Check GPU availability.
device = get_device_mode(args.gpu)
model, arch = get_model_arch(args.arch)
# Freeze the parameters of densenet.
for param in model.parameters():
    param.requires_grad = False
# Custom Classifier
classifier = nn.Sequential(
    nn.Linear(25088, args.hidden_uniits, bias=True),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(args.hidden_uniits, 102, bias=True),
    nn.LogSoftmax(dim=1),
)
model.classifier = classifier
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(),
                       lr=args.lr)
# Move to GPU or CPU mode
model = model.to(device)
with active_session():
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move inputs and labels to the GPU/CPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Set gradient to zero so that you do the parameter update correctly.
            # Else the gradient would point in some other direction
            # than the intended direction towards the minimum
            optimizer.zero_grad()
            # Propagation
            logps = model.forward(inputs)
            # Calculate loss
            loss = criterion(logps, labels)
            # Backpropagation
            loss.backward()
            # Update parameters based on the current gradient
            optimizer.step()
            running_loss += loss.item()
            # Print current error and accuracy in every n time
            if steps%print_every == 0:
                val_loss = 0
                val_accuracy = 0
                # Set to evaluation mode
                model.eval()
                # Deactivate autograde engine to save memory and time
                with torch.no_grad():
                    for inputs, labels in dataloaders['val']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()
                        # Since our model outputs a LogSoftmax, find the real
                        # percentages by reversing the log function
                        ps = torch.exp(logps)
                        # Get the top class and probability
                        top_p, top_class = ps.topk(1, dim=1)
                        # Check correct classes
                        equals = top_class == labels.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(dataloaders['train']))
                validation_losses.append(val_loss / len(dataloaders['val']))
                print(
                    f'Epoch {epoch+1}/{epochs}.. '
                    f'Train loss: {running_loss/print_every:.3f}.. '
                    f'Validation loss: {val_loss/len(dataloaders['val']):.3f}.. '
                    f'Validation accuracy: {val_accuracy/len(dataloaders['val']):.3f}'
                )
                running_loss = 0
                # Set to train mode
                model.train()

model.class_to_idx = data['train'].class_to_idx
checkpoint = {
    'input_size': 25088,
    'hidden_layers':[args.hidden_uniits],
    'output_size': 102,
    'arch': arch,
    'learning_rate': args.lr,
    'batch_size': 32,
    'classifier' : classifier,
    'epochs': epochs,
    'optimizer': optimizer.state_dict(),
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
}
filepath = args.save_dir + '/checkpoint.pth'
torch.save(checkpoint, filepath)
print('model saved')
