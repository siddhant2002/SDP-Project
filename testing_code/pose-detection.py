from transformers import pipeline
import torch
import torchvision
from PIL import Image
from datetime import datetime
import pymongo
from gridfs import GridFS
import os

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["siddhant2"]
collection = db["face_data"]
fs = GridFS(db)

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# Load the image classification pipeline
processor_name = "image-classification"
model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
pipe = pipeline(processor_name, model=model_name)

# Path to the Image1 folder
image_folder = r"C:\Users\siddh\Desktop\Final Year\test_images1"

# Initialize variables for previous action
prev_action = None
prev_face_data_id = None
action_map = {}  # Map actions to numbers
action_ids = {}  # Dictionary to store action labels and their most recent IDs

# Initialize the next action ID
next_action_id = 1

# Process each image in the Image1 folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        # Open and convert the image to RGB format
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        image_trans = transforms(image).unsqueeze(0).to('cuda')

        # Calculate image embeddings
        with torch.no_grad():
            image_embeddings = model(image_trans).squeeze().cpu().numpy()

        # Store the image embeddings in MongoDB using GridFS
        image_embeddings_bytes = image_embeddings.tobytes()
        embedding_filename = f'{filename.split(".")[0]}_embeddings'  # Use the filename as part of the embedding filename
        fs.put(image_embeddings_bytes, filename=embedding_filename)

        # Store the image and its embedding filename in a single document
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "timestamp": timestamp,
            "embedding_filename": embedding_filename,
            "image_path": image_path  # Optional: Store the path of the original image for reference
        }
        collection.insert_one(data)

        # Continue with the rest of your processing...
