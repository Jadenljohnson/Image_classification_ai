# predict_stdin.py
import sys
from io import BytesIO
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CKPT_PATH = "best_resnet18.pt"

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, classes

def preprocess_image_bytes(img_bytes: bytes):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return tfm(img).unsqueeze(0)

def main():
    img_bytes = sys.stdin.buffer.read()
    if not img_bytes:
        print("ERROR: no image bytes", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(CKPT_PATH, device)

    x = preprocess_image_bytes(img_bytes).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    pred = classes[idx]
    conf = float(probs[idx].item())
    print(f"{pred},{conf:.4f}")

if __name__ == "__main__":
    main()
