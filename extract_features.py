import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

IMAGE_DIR = "static/images"
FEATURES_FILE = "features.npy"
META_FILE = "meta.npy"

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

features = []
meta = []

for category in os.listdir(IMAGE_DIR):
    cat_path = os.path.join(IMAGE_DIR, category)
    if not os.path.isdir(cat_path):
        continue

    for img_name in os.listdir(cat_path):
        img_path = os.path.join(cat_path, img_name)
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(img).squeeze().numpy()

        features.append(feat)
        meta.append((category, img_name))

features = np.array(features).astype("float32")
np.save(FEATURES_FILE, features)
np.save(META_FILE, meta)

print("✅ Features + categories extracted!")
