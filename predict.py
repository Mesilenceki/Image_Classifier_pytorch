import argparse

import ison
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser= argparse.ArgumentParser()
parser.add_argument('-input', action = 'store', dest = 'image_path',
                    help = 'input a image to predict')
parser.add_argument('checkpoint', action = 'store', dest = 'model_path',
                    help= 'the path to load the previous model')
parser.add_argument('-top_k', action = 'store', dest = 'top_k',
                    help = "choose the str(top_k) result")
parser.add_argument('json_name', action = 'store', dest = 'category_name'
                    help = 'the json file to map the category_name')
parser.add_argument('-gpu', action = 'store', dest = 'device',
                    help = 'choice are cuda or cpu')

parameters = parser_args()

image_path = parameters.image_path
model_path = parameters.model_path
top_k = parameters.top_k
category_name = parameters.category_name
gpu = parameters.device

with open(category_name, 'r') as f:
    cat_to_name = json.load(f)

#preprocess the image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # open the image
    img = Image.open(image_path)
    #Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop    
    x0 = (img.width-224)/2
    y0 = (img.height-224)/2
    x1 = x0 + 224
    y1 = y0 + 224
    img = img.crop((x0, y0, x1,   y1))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))    
    return img
#load the previous model
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    #freeze the parameters of the model using params.require_grad = false
    model.classifier = checkpoint['classifier']     
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#predict the given picture                    
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use GPU if it's available
    device = torch.device(device)
    
    # Process image
    img = process_image(image_path)
    
    # Convert image from numpy to torch
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    # Add batch of size 1 to image
    torch_image = image_tensor.unsqueeze(0)    
    
    # Calculate accuracy
    loss = model.forward(torch_image)
    ps = torch.exp(loss)
    top_p, top_class = ps.topk(topk)
    top_p = top_p.detach().tolist()[0] 
    top_class = top_class.detach().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_flowers = [cat_to_name[idx_to_class[label]] for label in top_class]
    topk_class = []
    for index in top_class:
        topk_class += [idx_to_class[index]]
    return top_p, topk_class, top_flowers

def main():
    img = process_image(image_path)
    model = load_check(model_path)
    top_prob,topk_class,top_flowers = predict(img, model, top_k, gpu)
    print("Top 5 predict class is {}".format(topk_class))
    print("Top 5 predict flower name is{}".format(top_flowers))
    print("Their probability is{}".format(top_prob))
    pass
if __name__ == '__main__':
    main()