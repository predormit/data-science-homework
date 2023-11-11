import torch

import torchvision.transforms as transforms

from PIL import Image

 # Read the image

picture = Image.open('saber.jpg')

 # Define transform

transform = transforms.Grayscale()

 # Convert the image to grayscale

image = transform(picture)

 # Display

image.show()