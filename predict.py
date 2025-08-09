# predict.py
import sys
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

# Same normalization used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_model(num_classes: int):
    # Same architecture as training
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]  # ['cat', 'dog']
    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, classes

def preprocess_image(image_path: str):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)  # Add batch dimension

def predict(image_path: str, ckpt_path="best_resnet18.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(ckpt_path, device)

    img_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = classes[pred_idx]
        confidence = probs[0][pred_idx].item()

    print(f"Prediction: {pred_class} ({confidence:.4f} confidence)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found.")
        sys.exit(1)
    predict(image_path)
