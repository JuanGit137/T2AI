import torch
from torchvision import transforms, models
from PIL import Image

if __name__ == '__main__':            
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
    # load de model 
    # you can try with resnet18 or resenet50 as well.
    model = models.resnet34(pretrained=True).to(device)
    model.fc = torch.nn.Identity() #cancel the classification layer
    
    image_path = "images/example_2.jpg"
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # add an extra dimension for batch
    #Pasamos la imagen por el modelo
    with torch.no_grad():
        features = model(image)
        dim = features.shape[1]                                
    print('dim = {}'.format(dim))            