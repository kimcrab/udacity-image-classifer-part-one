import numpy as np
import torchvision
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from PIL import Image

from arguments import get_prediction_args
from utils import get_device_mode, load_checkpoint, map_categories
from data_processor import process_image

parser = get_prediction_args()
args = parser.parse_args()
nn_filename = args.checkpoint
# Load saved model
model, optimizer = load_checkpoint(nn_filename)
# Move to GPU or CPU mode
device = get_device_mode(args.gpu)
model = model.to(device)
img_path = args.img_dir
topk = args.top_k

image = process_image(args.img_dir)
image = torch.from_numpy(np.array([image])).float()
image = Variable(image).to(device)

model.eval()
with torch.no_grad():
    logps = model.forward(image)
    ps = torch.exp(logps).cpu()
    probs, labels = torch.topk(ps, topk)
    class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = []
    for label in labels.numpy()[0]:
        classes.append(class_to_idx_rev[label])
pre_classes = classes
probability = probs.numpy()[0]

if args.category_file is not None:
    cat_to_name = map_categories(args.category_file)
    print(probability)
    print([cat_to_name[x] for x in pre_classes])
else:
    print(probability)
    print(pre_classes)
