from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

Data_dir = "animals"
Img_size = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
tfms = transforms.Compose([
    transforms.Resize((Img_size,Img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_data_set = datasets.ImageFolder(f"{Data_dir}/train",transform=tfms)
train_loader = DataLoader(train_data_set, batch_size=8, shuffle=True)
print("Classes found:", train_data_set.classes)
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels.tolist())
plt.figure(figsize=(10, 6))
for i in range(len(images)):
    img = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(train_data_set.classes[labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()