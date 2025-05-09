import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os

# Corregimos las rutas para tu configuración local
DATASET = 'simple1K'
MODEL = 'resnet34'
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', 'simple1K')
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')

# Creamos el directorio data si no existe
os.makedirs('data', exist_ok=True)

if __name__ == '__main__':
    print(f"Buscando imágenes en: {image_dir}")
    print(f"Buscando lista de imágenes en: {list_of_images}")
    
    try:
        #reading data
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
        print(f"Se encontraron {len(files)} imágenes en la lista")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {list_of_images}")
        print("Verifica que la estructura de carpetas sea correcta:")
        print(f"- {data_dir}")
        print(f"  └── images/")
        print(f"  └── list_of_images.txt")
        exit(1)
        
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # defining the image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    
    #load the model with updated parameter   
    model = None
    if MODEL == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1').to(device)
        model.fc = torch.nn.Identity() 
    if MODEL == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1').to(device)
        model.fc = torch.nn.Identity() 
    #you can add more models

    dim = 512
    #Pasamos la imagen por el modelo
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype=np.float32)        
        
        for i, file in enumerate(files):
            try:                
                filename = os.path.join(image_dir, file[0])
                print(f"Procesando: {filename}")
                image = Image.open(filename).convert('RGB')
                image = preprocess(image).unsqueeze(0).to(device)
                features[i,:] = model(image).cpu()[0,:]
                
                if i%20 == 0 and i > 0:
                    print(f'Procesadas {i}/{n_images} imágenes')
            except Exception as e:
                print(f"ERROR al procesar {filename}: {e}")
                
        feat_file = os.path.join('data', f'feat_{MODEL}_{DATASET}.npy')
        np.save(feat_file, features)
        print(f'Características guardadas en {feat_file}')