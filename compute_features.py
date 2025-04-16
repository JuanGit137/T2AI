import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os

# data_dir = '/hd_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# image_dir = os.path.join(data_dir, 'JPEGImages')
# val_file = 'data/voc_val.txt'
# data_dir = '/hd_data/Paris/'
# image_dir = os.path.join(data_dir, 'paris')
# val_file = 'data/val_paris.txt'
DATASET = 'simple1k'
MODEL = 'resnet34'
data_dir = '/hd_data/simple1K/'
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')
if __name__ == '__main__':
    #reading data
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # defining the image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    #load de model     
    model = None
    if MODEL == 'resnet18' :
        model = models.resnet18(pretrained=True).to(device)
        model.fc = torch.nn.Identity() 
    if MODEL == 'resnet34' :
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity() 
    #you can add more models

    dim = 512
    #Pasamos la imagen por el modelo
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype = np.float32)        
        for i, file in enumerate(files) :                
            filename = os.path.join(image_dir, file[0])
            image = Image.open(filename).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            features[i,:] = model(image).cpu()[0,:]
            if i%100 == 0 :
                print('{}/{}'.format(i, n_images))            
                
        feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
        np.save(feat_file, features)
        print('saving data ok')