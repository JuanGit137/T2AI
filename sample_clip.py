#https://github.com/openai/CLIP?tab=readme-ov-file#usage

import clip
#required-> pip install git+https://github.com/openai/CLIP.git
import torch
from torchvision import transforms, models
from PIL import Image

if __name__ == '__main__':            
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    # load de model 
    model, preprocess_clip = clip.load("ViT-B/32", device=device)
    model = model.encode_image    
    image_path = "images/example_2.jpg"
    image = Image.open(image_path).convert('RGB')
    image = preprocess_clip(image).unsqueeze(0).to(device)  # add an extra dimension for batch
    #Pasamos la imagen por el modelo
    with torch.no_grad():
        features = model(image)
        dim = features.shape[1]                                
    print('dim = {}'.format(dim))            