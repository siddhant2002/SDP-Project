import os
import torch
import torchvision
from PIL import Image
from datetime import datetime
import pymongo
from gridfs import GridFS

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["siddhant"]
collection = db["face_data"]
fs = GridFS(db)

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# Define the path to the folder containing images
folder_path = r"C:\Users\siddh\Desktop\Final Year\test_images1"

# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open and convert the image to RGB format
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
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

        # Store the timestamp and embedding filename in a single document
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "timestamp": timestamp,
            "embedding_filename": embedding_filename,
        }
        collection.insert_one(data)

        # Print the timestamp and confirmation message for each image
        print(f"Timestamp for {filename}: {timestamp}")
        print(f"Image embeddings for {filename} stored in MongoDB")

# The fs.chunks and fs.files collections are part of GridFS, which is used by MongoDB to store large files, such as the image embeddings you are saving. The fs.files collection contains metadata about the files stored in GridFS, while the fs.chunks collection contains the actual binary data of the files, divided into chunks.

# When you store a file using GridFS, MongoDB automatically creates these collections to manage the storage and retrieval of the file's data. They are essential for GridFS to function correctly, so there's no need to worry about them.