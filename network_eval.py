#! /usr/bin/env python3

# Imports
import sys
import torch
import torchvision.transforms as transforms
from glob import glob
import os
from os.path import join, basename
from PIL import Image
import json
from gdrive_downloader import download_file_from_google_drive

# Argument parsing
argv = sys.argv[1:]
if len(argv) != 1:
    raise Exception("Wrong number of arguments passed")

# Constants
CLASSES = {0: 'Female', 1:'Male'}
IMG_FOLDER = argv[0]
CHECKPOINT_PATH = 'state.pth'
CHECKPOINT_FILE_ID = '1--wMPNZBgBV4pEcBEDtOajRotm1LWb5N'

# Transform
init_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# Get list of images in folder
image_paths = sorted(glob(join(IMG_FOLDER, '*.jpg')))

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load network and weights
from torchvision.models import vgg16
net = vgg16(pretrained=False, num_classes=2)
if not os.path.exists(CHECKPOINT_PATH):
    download_file_from_google_drive(CHECKPOINT_FILE_ID, CHECKPOINT_PATH)

net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))


# Process images
results = {}
for i in image_paths:
    img = init_transform(Image.open(i))
    img.unsqueeze_(0)
    with torch.no_grad():
        output = net(img)
        _, predicted = torch.max(output.data, 1)
        results[basename(i)] = CLASSES[predicted.tolist()[0]]

# Generate JSON file
with open("process_results.json", "w") as write_file:
    json.dump(results, write_file, indent=0)
