# import cv2
# from ultralytics import YOLO
# import os, torch

# print(torch.cuda.is_available())
# # Load a model
# model = YOLO('yolov8l.pt').to('cuda')

# # # Train the model
# results = model.track(source=0, show=True, tracker="botsort.yaml")

# action recognition

# from transformers import pipeline
# from PIL import Image

# pipe = pipeline("image-classification", "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224").to('cuda')
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11.jpg").convert("RGB")
# print(pipe(image))


from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224", device=0)  # Assuming 0 is the CUDA device index
image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11.jpg").convert("RGB")
print(pipe(image))



# image embedding
import torch
import numpy as np
import torchvision
from PIL import Image
from scipy.spatial.distance import cdist

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')
image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11.jpg").convert("RGB")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    ),
])

image_trans = transforms(image).unsqueeze(0).to('cuda')

with torch.no_grad():
    image_embeddings = model(image_trans).squeeze().cpu().numpy()

all_other_embeddings = np.stack([image_embeddings, image_embeddings, image_embeddings, image_embeddings])
pairwise_dist = cdist(image_embeddings.reshape(1, -1), all_other_embeddings.reshape(-1, 1000))
best_match_idx = np.argmin(pairwise_dist)
print(pairwise_dist)

# ts, y