import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import time
from torch.hub import load_state_dict_from_url
from tqdm import tqdm

# Intenta importar CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP no está instalado. Para instalarlo, ejecuta: pip install git+https://github.com/openai/CLIP.git")

# Lista de datasets y modelos a procesar
DATASETS = ['simple1K', 'Paris_val', 'VOC_val']
MODELS = ['resnet18', 'resnet34', 'dinov2', 'clip']

# Creamos el directorio data si no existe
os.makedirs('data', exist_ok=True)

# Configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    """
    Carga el modelo especificado y configura su salida para extracción de características
    """
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
        # Cargar DINOv2 small model
        try:
            # Asegurarnos de no importar localmente
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            dim = 384  # Dimensión de las características para vits14
        except Exception as e:
            print(f"Error cargando DINOv2: {e}")
            print("Intentando método alternativo...")
            try:
                # Método alternativo usando load_state_dict_from_url
                url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False).to(device)
                state_dict = load_state_dict_from_url(url)
                model.load_state_dict(state_dict)
                dim = 384
            except Exception as e:
                print(f"Error con método alternativo: {e}")
                return None, 0, None
    
    elif model_name == 'clip':
        if not CLIP_AVAILABLE:
            print("CLIP no está disponible. Saltando este modelo.")
            return None, 0, None
        
        try:
            model, preprocess_clip = clip.load("ViT-B/32", device=device)
            dim = 512  # Dimensión de las características para CLIP ViT-B/32
            # Retornamos también el preprocesamiento específico para CLIP
            return model, dim, preprocess_clip
        except Exception as e:
            print(f"Error cargando CLIP: {e}")
            print("Asegúrate de tener instalado CLIP con: pip install git+https://github.com/openai/CLIP.git")
            return None, 0, None
    
    if model_name != 'clip':
        return model, dim, None
    
    return None, 0, None

def get_preprocessing(model_name, clip_preprocess=None):
    """
    Retorna la función de preprocesamiento adecuada para cada modelo
    """
    if model_name == 'clip' and clip_preprocess is not None:
        return clip_preprocess
    
    # Preprocesamiento estándar para ResNet y otros modelos
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
    """
    Procesa las imágenes del dataset especificado utilizando el modelo proporcionado
    """
    # Configurar rutas
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', dataset)
    image_dir = os.path.join(data_dir, 'images')
    list_of_images = os.path.join(data_dir, 'list_of_images.txt')
    
    print(f"\nProcesando dataset {dataset} con modelo {model_name}")
    print(f"Buscando imágenes en: {image_dir}")
    print(f"Buscando lista de imágenes en: {list_of_images}")
    
    try:
        # Leer datos
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
        print(f"Se encontraron {len(files)} imágenes en la lista")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {list_of_images}")
        print("Verifica que la estructura de carpetas sea correcta:")
        print(f"- {data_dir}")
        print(f"  └── images/")
        print(f"  └── list_of_images.txt")
        return False
    
    # Dimensión de las características
    if model_name == 'dinov2':
        dim = 384
    elif model_name in ['resnet18', 'resnet34', 'clip']:
        dim = 512
    else:
        dim = 512  # Valor predeterminado
    
    start_time = time.time()
    
    # Extraer características
    with torch.no_grad():
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype=np.float32)
        
        # Usar tqdm para mostrar una barra de progreso
        for i, file in enumerate(tqdm(files, desc=f"Procesando {model_name} - {dataset}")):
            try:
                image_path = file[0]
                # Manejar rutas con formato clase/imagen.jpg
                if '/' in image_path:
                    image_path = os.path.join(*image_path.split('/'))
                
                filename = os.path.join(image_dir, image_path)
                
                image = Image.open(filename).convert('RGB')
                preprocessed_image = preprocess(image).unsqueeze(0).to(device)
                
                if model_name == 'clip':
                    # CLIP tiene un método específico para extraer características
                    features[i,:] = model.encode_image(preprocessed_image).cpu().numpy()[0]
                else:
                    features[i,:] = model(preprocessed_image).cpu().numpy()[0]
                
            except Exception as e:
                print(f"\nERROR al procesar {filename}: {e}")
        
        # Guardar características
        feat_file = os.path.join('data', f'feat_{model_name}_{dataset}.npy')
        np.save(feat_file, features)
        
        elapsed_time = time.time() - start_time
        print(f'Características guardadas en {feat_file}')
        print(f'Tiempo de procesamiento: {elapsed_time:.2f} segundos')
    
    return True

def main():
    # Asegurarse de que exista la carpeta 'data'
    os.makedirs('data', exist_ok=True)
    
    print(f"Usando dispositivo: {device}")
    
    # Procesar cada combinación de modelo y dataset
    for model_name in MODELS:
        # Cargar el modelo
        if model_name == 'clip':
            if not CLIP_AVAILABLE:
                print("CLIP no está disponible. Saltando el modelo CLIP.")
                continue
                
            model, dim, preprocess_clip = get_model(model_name)
            if model is None:
                print(f"No se pudo cargar el modelo {model_name}. Saltando...")
                continue
            preprocess = preprocess_clip
        else:
            model, dim, _ = get_model(model_name)
            if model is None:
                print(f"No se pudo cargar el modelo {model_name}. Saltando...")
                continue
            preprocess = get_preprocessing(model_name)
        
        # Configurar el modelo para evaluación
        model.eval()
        
        for dataset in DATASETS:
            success = process_images(model, model_name, dataset, preprocess)
            if not success:
                print(f"Hubo un problema al procesar {dataset} con {model_name}")
            
            # Liberar memoria CUDA si es posible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print("\n¡Proceso completo! Todas las características han sido extraídas.")

if __name__ == '__main__':
    main()