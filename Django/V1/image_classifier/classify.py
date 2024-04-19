#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Imports here
import time
inicio = time.time()
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

# Data folders
data_dir = 'data60X'
valid_dir = data_dir + '/fotos'

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Check the contents of cat_to_name
for i, key in enumerate(cat_to_name.keys()):
    print(key, '\t->', cat_to_name[key])
    #if i == 10:
        #break
        
print("There are {} image categories.".format(len(cat_to_name)))

# Building and training the classifier
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x

# Loading the checkpoint
def rebuildModel(filepath):
    
    # Load model metadata
    # Loading weights for CPU model whoch were trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model

filename = 'model_20230721_171116-9655.pth'
# TODO: Write a function that loads a checkpoint and rebuilds the model

# Classifier class definition must be loaded (if not done) 
# in order to rebuild the model (see above)

model_from_file = rebuildModel(filename)
#model_from_file

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    
    # Find the shorter side and resize it to 256 keeping aspect ration
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        plt.title(title)
    
    ax.imshow(image)
    
    return ax
cat_to_name.json
# Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


def convertCategoryToName(categories, mapper='cat_to_name.json'):
    
    # Load json file
    with open(mapper, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
    
        names = []

        # Find flower names corresponding to predicted categories
        for category in categories:
            names.append(cat_to_name[str(category)])

    return names

# Select random image and predict its class
# Visualize the result
def display_preds(path, model, topk=5, maderas_names=cat_to_name):
    image_path = path

    # Return random path to an image
    #image_path = select_random_image(path)
    image_path = path.replace('\\', '/')
    print(image_path)
    
    # Return folder number which equals to 
    # a class identifier
    #folder_number = image_path.split('/')[8]
    #print(folder_number)
    
    # Read the wood name based on the folder
    # number (wood class id) and external dictionary mapping
    #title = maderas_names[folder_number]
    #print(title)
    
    # Predict image class
    probs, classes = predict(image_path, model, topk)
    #print(probs)
    #print(classes)
    
    # Convert class id into its name
    names = convertCategoryToName(classes)
    print('n:', names)
    print('c:', classes)
    title = names[0]
    # Open an image
    image = Image.open(image_path)
    
    # Make the image compliant with PyTorch
    image = process_image(image)
    
    # Set up a plot
    plt.figure(figsize = (6, 10))
    ax = plt.subplot(2, 1, 1)

    # Plot the flower
    imshow(image, ax, title=title);

    # Visualize predictivalidationon result
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);
    plt.show()

    return

dirname = os.path.join(os.getcwd(), valid_dir) Imports here
import time
inicio = time.time()
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

# Data folders
data_dir = 'data60X'
valid_dir = data_dir + '/fotos'

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Check the contents of cat_to_name
for i, key in enumerate(cat_to_name.keys()):
    print(key, '\t->', cat_to_name[key])
    #if i == 10:
        #break
        
print("There are {} image categories.".format(len(cat_to_name)))

# Building and training the classifier
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x

# Loading the checkpoint
def rebuildModel(filepath):
    
    # Load model metadata
    # Loading weights for CPU model whoch wer Imports here
import time
inicio = time.time()
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

# Data folders
data_dir = 'data60X'
valid_dir = data_dir + '/fotos'

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Check the contents of cat_to_name
for i, key in enumerate(cat_to_name.keys()):
    print(key, '\t->', cat_to_name[key])
    #if i == 10:
        #break
        
print("There are {} image categories.".format(len(cat_to_name)))

# Building and training the classifier
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x

# Loading the checkpoint
def rebuildModel(filepath):
    
    # Load model metadata
    # Loading weights for CPU model whoch were trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model

filename = 'model_20230721_171116-9655.pth'
# TODO: Write a function that loads a checkpoint and rebuilds the model

# Classifier class definition must be loaded (if not done) 
# in order to rebuild the model (see above)

model_from_file = rebuildModel(filename)
#model_from_file

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    
    # Find the shorter side and resize it to 256 keeping aspect ration
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        plt.title(title)
    
    ax.imshow(image)
    
    return ax
cat_to_name.json
# Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


def convertCategoryToName(categories, mapper='cat_to_name.json'):
    
    # Load json file
    with open(mapper, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
    
        names = []

        # Find flower names corresponding to predicted categories
        for category in categories:
            names.append(cat_to_name[str(category)])

    return names

# Select random image and predict its class
# Visualize the result
def display_preds(path, model, topk=5, maderas_names=cat_to_name):
    image_path = path

    # Return random path to an image
    #image_path = select_random_image(path)
    image_path = path.replace('\\', '/')
    print(image_path)
    
    # Return folder number which equals to 
    # a class identifier
    #folder_number = image_path.split('/')[8]
    #print(folder_number)
    
    # Read the wood name based on the folder
    # number (wood class id) and external dictionary mapping
    #title = maderas_names[folder_number]
    #print(title)
    
    # Predict image class
    probs, classes = predict(image_path, model, topk)
    #print(probs)
    #print(classes)
    
    # Convert class id into its name
    names = convertCategoryToName(classes)
    print('n:', names)
    print('c:', classes)
    title = names[0]
    # Open an image
    image = Image.open(image_path)
    
    # Make the image compliant with PyTorch
    image = process_image(image)
    
    # Set up a plot
    plt.figure(figsize = (6, 10))
    ax = plt.subplot(2, 1, 1)

    # Plot the flower
    imshow(image, ax, title=title);

    # Visualize predictivalidationon result
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);
    plt.show()

    return

dirname = os.path.join(os.getcwd(), valid_dir)
imgpath = dirname + os.sep 

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            display_preds(filepath,model_from_file,maderas_names=cat_to_name)

fin = time.time()
print(fin-inicio)e trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model

filename = 'model_20230721_171116-9655.pth'
# TODO: Write a function that loads a checkpoint and rebuilds the model

# Classifier class definition must be loaded (if not done) 
# in order to rebuild the model (see above)

model_from_file = rebuildModel(filename)
#model_from_file

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    
    # Find the shorter side and resize it to 256 keeping aspect ration
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        plt.title(title)
    
    ax.imshow(image)
    
    return ax
cat_to_name.json
# Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


def convertCategoryToName(categories, mapper='cat_to_name.json'):
    
    # Load json file
    with open(mapper, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
    
        names = []

        # Find flower names corresponding to predicted categories
        for category in categories:
            names.append(cat_to_name[str(category)])

    return names

# Select random image and predict its class
# Visualize the result
def display_preds(path, model, topk=5, maderas_names=cat_to_name):
    image_path = path

    # Return random path to an image
    #image_path = select_random_image(path)
    image_path = path.replace('\\', '/')
    print(image_path)
    
    # Return folder number which equals to 
    # a class identifier
    #folder_number = image_path.split('/')[8]
    #print(folder_number)
    
    # Read the wood name based on the folder
    # number (wood class id) and external dictionary mapping
    #title = maderas_names[folder_number]
    #print(title)
    
    # Predict image class
    probs, classes = predict(image_path, model, topk)
    #print(probs)
    #print(classes)
    
    # Convert class id into its name
    names = convertCategoryToName(classes)
    print('n:', names)
    print('c:', classes)
    title = names[0]
    # Open an image
    image = Image.open(image_path)
    
    # Make the image compliant with PyTorch
    image = process_image(image)
    
    # Set up a plot
    plt.figure(figsize = (6, 10))
    ax = plt.subplot(2, 1, 1)

    # Plot the flower
    imshow(image, ax, title=title);

    # Visualize predictivalidationon result
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);
    plt.show()

    return

dirname = os.path.join(os.getcwd(), valid_dir)
imgpath = dirname + os.sep 

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            display_preds(filepath,model_from_file,maderas_names=cat_to_name)

fin = time.time()
print(fin-inicio)
imgpath = dirname + os.sep 

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            display_preds(filepath,model_from_file,maderas_names=cat_to_name)

fin = time.time()
print(fin-inicio)
            


# In[ ]:





# In[ ]:




