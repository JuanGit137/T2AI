import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import time
from torch.hub import load_state_dict_from_url
from tqdm import tqdm

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

DATASETS = ['simple1K', 'Paris_val', 'VOC_val']
MODELS = ['resnet18', 'resnet34', 'dinov2', 'clip']

os.makedirs('data', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    model = None
    dim = 0
    
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1').to(device)
        model.fc = torch.nn.Identity()
        dim = 512
    
    elif model_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1').to(device)
        model.fc = torch.nn.Identity()
        dim = 512
    
    elif model_name == 'dinov2':
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            dim = 384
        except Exception:
            try:
                url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False).to(device)
                state_dict = load_state_dict_from_url(url)
                model.load_state_dict(state_dict)
                dim = 384
            except Exception:
                return None, 0, None
    
    elif model_name == 'clip':
        if not CLIP_AVAILABLE:
            return None, 0, None
        
        try:
            model, preprocess_clip = clip.load("ViT-B/32", device=device)
            dim = 512
            return model, dim, preprocess_clip
        except Exception:
            return None, 0, None
    
    if model_name != 'clip':
        return model, dim, None
    
    return None, 0, None

def get_preprocessing(model_name, clip_preprocess=None):
    if model_name == 'clip' and clip_preprocess is not None:
        return clip_preprocess
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

def process_images(model, model_name, dataset, preprocess):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', dataset)
    image_dir = os.path.join(data_dir, 'images')
    list_of_images = os.path.join(data_dir, 'list_of_images.txt')
    
    try:
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
    except FileNotFoundError:
        return False
    
    if model_name == 'dinov2':
        dim = 384
    elif model_name in ['resnet18', 'resnet34', 'clip']:
        dim = 512
    else:
        dim = 512
    
    with torch.no_grad():
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype=np.float32)
        
        for i, file in enumerate(files):
            try:
                image_path = file[0]
                if '/' in image_path:
                    image_path = os.path.join(*image_path.split('/'))
                
                filename = os.path.join(image_dir, image_path)
                
                image = Image.open(filename).convert('RGB')
                preprocessed_image = preprocess(image).unsqueeze(0).to(device)
                
                if model_name == 'clip':
                    features[i,:] = model.encode_image(preprocessed_image).cpu().numpy()[0]
                else:
                    features[i,:] = model(preprocessed_image).cpu().numpy()[0]
                
            except Exception:
                pass
        
        feat_file = os.path.join('data', f'feat_{model_name}_{dataset}.npy')
        np.save(feat_file, features)
    
    return True

def main():
    os.makedirs('data', exist_ok=True)
    
    for model_name in MODELS:
        if model_name == 'clip':
            if not CLIP_AVAILABLE:
                continue
                
            model, dim, preprocess_clip = get_model(model_name)
            if model is None:
                continue
            preprocess = preprocess_clip
        else:
            model, dim, _ = get_model(model_name)
            if model is None:
                continue
            preprocess = get_preprocessing(model_name)
        
        model.eval()
        
        for dataset in DATASETS:
            process_images(model, model_name, dataset, preprocess)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()