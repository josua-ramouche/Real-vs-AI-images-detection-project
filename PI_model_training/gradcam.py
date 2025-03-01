import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision import models
from PIL import Image
from codecarbon import EmissionsTracker
import sys
import torch


state_dict = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=True)

model = models.resnet50() 
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features,2)
model.load_state_dict(state_dict) 
model.eval()

from torchvision import transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # Charger l'image et l'appliquer
    image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image.size


# Extraire les activations et les gradients du dernier bloc convolutif
def register_hooks(model):
    gradients = []
    activations = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, input, output):
        gradients.append(output[0])

    # Trouver la dernière couche convolutive
    target_layer = model.layer4[2].conv3
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)

    return target_layer, gradients, activations

# Fonction pour calculer la heatmap Grad-CAM
def generate_gradcam(gradients, activations, image_tensor):

    gradient = gradients[0].detach().cpu().numpy()
    activation = activations[0].detach().cpu().numpy()

    weights = np.mean(gradient, axis=(2, 3))[0, :]  # Moyenne sur H et W (pour chaque canal)
    
    # Calculer la carte de Grad-CAM en multipliant les activations par les poids
    grad_cam_map = np.sum(activation[0] * weights[:, np.newaxis, np.newaxis], axis=0)

    # Vérifier que np.max() et np.min() ne sont pas égaux
    grad_cam_map_min = np.min(grad_cam_map)
    grad_cam_map_max = np.max(grad_cam_map)

    if grad_cam_map_max - grad_cam_map_min != 0:
        grad_cam_map = (grad_cam_map - grad_cam_map_min) / (grad_cam_map_max - grad_cam_map_min)
    else:
        grad_cam_map = np.zeros_like(grad_cam_map)  # Si la gamme est nulle, mettre la carte à zéro

    grad_cam_map = cv2.resize(grad_cam_map, (image_tensor.shape[2], image_tensor.shape[3]))  # Redimensionner à la taille d'entrée
    grad_cam_map = (grad_cam_map - np.min(grad_cam_map)) / (np.max(grad_cam_map) - np.min(grad_cam_map))  # Normaliser

    return grad_cam_map


def show_gradcam(image, grad_cam_map, target_class):

    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_cv2, (grad_cam_map.shape[1], grad_cam_map.shape[0]))

    # Application de la carte thermique Grad-CAM à l'image
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map), cv2.COLORMAP_JET)  # Appliquer une carte thermique
    heatmap = np.float32(heatmap) / 255
    gradcam_overlay = heatmap + np.float32(image_resized) / 255  # Ajouter la carte thermique à l'image

    gradcam_overlay = gradcam_overlay / np.max(gradcam_overlay)

    gradcam_overlay = np.uint8(255 * gradcam_overlay)
    if target_class == 0 : class_name = "FAKE" 
    else : class_name = "REAL"
    label = f'Predicted Class: {class_name}'  
    
    return gradcam_overlay


def run_gradcam(image):
    image_tensor, image_size = preprocess_image(image)
    target_layer, gradients, activations = register_hooks(model)

    output = model(image_tensor)
   
    pred_probs = torch.softmax(output, dim=1)
    
    predicted_class = pred_probs.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, predicted_class].backward()
    
    grad_cam_map = generate_gradcam(gradients, activations, image_tensor)

    grad_cam = show_gradcam(image, grad_cam_map, predicted_class)
    return cv2.resize(grad_cam, image_size)


def main() -> None:
    img = Image.open(sys.argv[1])

    global_tracker = EmissionsTracker(
    measure_power_secs=10, allow_multiple_runs=True)
    global_tracker.start()

    gradcam = run_gradcam(img)

    total_emissions = global_tracker.stop()
    print(f"Total emissions for training: {total_emissions:.8f} kg CO2eq")
    
    plt.imshow(gradcam)
    plt.axis(False)
    plt.show()


def usage() -> None:
    """ usage():
        Prints the usage instructions for the script. """
    print(f"Usage: python {sys.argv[0]} images/<image>")
    
    
if __name__ == '__main__':
    img_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    if len(sys.argv) != 2:
        print("Invalid number of arguments.")
        usage()
    if f".{sys.argv[1].split('.')[1]}" not in img_extensions:
        print("Invalid image path.")
        usage()
    else:
        main()


