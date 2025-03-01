""" 
This script provides functionality to load a pre-trained PyTorch model and use it to make predictions on a given image.
Usage:
    python prediction.py <model_path>.pt <image_path>
Example:
    python prediction.py out/model.pt images/sample.jpg
"""
import sys
import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore


def predict(model: torch.nn.Module,
            img: Image,
            transform: transforms) -> tuple:
    """ predict(model: torch.nn.Module, img: Image, transform: transforms) -> Tuple[int, float]:
        Given a model, an image, and a transform, this function returns the predicted class and confidence score. """
    img_jpg = img.convert("RGB")
    img_tensor = transform(img_jpg)
    img_tensor_batch = img_tensor.unsqueeze(dim=0)
    model.cpu().eval()
    with torch.inference_mode():
        y_logits = model(img_tensor_batch).squeeze()
        y_pred_probs = y_logits.softmax(dim=0)
        y_pred = y_pred_probs.argmax(dim=0).item()
        confidence = torch.max(y_pred_probs)
    return int(y_pred), float(confidence)


def main() -> None:
    """ main():
        Main function to load the model and image, apply transformations, and print the prediction and confidence. """    
    classes = ['FAKE', 'REAL']
    model = torch.load(sys.argv[1])
    img = Image.open(sys.argv[2])

    img_transform_size = model.input_shape[1:]
    
    img_transform = transforms.Compose([
        transforms.Resize(size=img_transform_size),
        transforms.ToTensor()
    ])
    pred, confidence = predict(model, img, img_transform)
    print(f"This image is {classes[pred].lower()}!")
    print(f"Confidence: {confidence*100:.2f}%")

def usage() -> None:
    """ usage():
        Prints the usage instructions for the script. """
    print(f"Usage: python {sys.argv[0]} out/<model_name>.pt images/<image>")
    
    
if __name__ == '__main__':
    img_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    if len(sys.argv) != 3:
        print("Invalid number of arguments.")
        usage()
    if not (sys.argv[1].endswith(".pt") or sys.argv[1].endswith(".pth")):
        print("Invalid model path.")
        usage()
    if f".{sys.argv[2].split('.')[1]}" not in img_extensions:
        print("Invalid image path.")
        usage()
    else:
        main()