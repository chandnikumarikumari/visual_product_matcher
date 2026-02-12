from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import io
import requests

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ResNet model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load features, metadata & FAISS index
features = np.load("features.npy")
meta = np.load("meta.npy", allow_pickle=True)
index = faiss.read_index("index.faiss")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    file: UploadFile = File(None),
    image_url: str = Form(None)
):
    # Load image from file or URL
    if file:
        image_bytes = await file.read()
    elif image_url:
        r = requests.get(image_url)
        image_bytes = r.content
    else:
        return templates.TemplateResponse("index.html", {"request": request, "results": None})

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img).squeeze().numpy().astype("float32")

    # Search FAISS
    D, I = index.search(np.array([feat]), 12)

    # 🔥 Category Filtering Logic
    best_category = meta[I[0][0]][0]

    results = []
    for idx, dist in zip(I[0], D[0]):
        category, filename = meta[idx]

        if category != best_category:
            continue

        results.append({
            "image": f"/static/images/{category}/{filename}",
            "score": round(float(1 / (1 + dist)), 4)
        })

        if len(results) == 6:
            break

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results
    })
