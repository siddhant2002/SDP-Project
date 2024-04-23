# import torch
# import torchvision
# from PIL import Image
# from gridfs import GridFS
# import pymongo
# import numpy as np
# import os

# # Connect to MongoDB
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["siddhant1"]
# fs = GridFS(db)

# # Load the ResNet50 model
# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# # Function to extract face embedding data from a new image
# def extract_face_embedding(image_path):
#     # Open and convert the image to RGB format
#     image = Image.open(image_path).convert("RGB")

#     # Preprocess the image
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize((224, 224)),  # Resize to match ResNet input size
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#     ])
#     image_trans = transforms(image).unsqueeze(0).to('cuda')

#     # Calculate image embeddings
#     with torch.no_grad():
#         image_embeddings = model(image_trans).squeeze().cpu().numpy()

#     # Convert image embeddings to bytes
#     image_embeddings_bytes = image_embeddings.tobytes()
#     return image_embeddings_bytes

# # Function to check if face embedding data exists in the database
# import os

# def check_face_embedding(image_embeddings_bytes, path):
#     # Convert filename to bytes
#     filename_bytes = os.path.basename(path)
#     embedding_filename = f'{filename_bytes.split(".")[0]}_embeddings'  # Use the filename as part of the embedding filename
#     print(embedding_filename)
#     # Check if the face embedding data exists in the GridFS collection
#     existing_data = list(fs.find({"filename": embedding_filename}))
#     existing_data_count = len(existing_data)
#     if existing_data_count > 0:
#         # Face embedding data already exists, retrieve its document ID
#         face_data_id = existing_data[0]._id
#         # Retrieve the image embeddings from GridFS
#         image_embeddings = fs.get(face_data_id).read()
#     else:
#         # Face embedding data does not exist, return None
#         face_data_id = None
#         image_embeddings = None
#     return face_data_id, image_embeddings




# # # Example usage
# # image_path = "C:/Users/siddh/Desktop/Final Year/test_images1/frame19_pre.jpg"
# # image_embeddings_bytes = extract_face_embedding(image_path)
# # # print(image_embeddings_bytes)
# # face_data_id = check_face_embedding(image_embeddings_bytes,image_path)
# # print(face_data_id)
# # if face_data_id is not None:
# #     print("Face embedding data ID:", face_data_id)
# # else:
# #     print("Face embedding data does not exist in the database.")




# # Example usage
# image_path = "C:/Users/siddh/Desktop/Final Year/test_images1/frame19_pre.jpg"
# image_embeddings_bytes = extract_face_embedding(image_path)
# # print(image_embeddings_bytes)
# face_data_id, image_embeddings = check_face_embedding(image_embeddings_bytes,image_path)
# print(face_data_id)
# if face_data_id is not None:
#     print("Face embedding data ID:", face_data_id)
#     # print("Image embeddings:", image_embeddings)
#     print("found")
# else:
#     print("Face embedding data does not exist in the database.")

import torch
import torchvision
import numpy as np
from PIL import Image
import pymongo
from scipy.spatial.distance import cdist

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["siddhant1"]
collection = db["face_data"]

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# Path to the Image1 folder
image_path = r"C:/Users/siddh/Desktop/Final Year/Images/frame11_pre.jpg"

# Preprocess the image
transforms = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(), 
torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
    ),
])

people_embeddings_arr = []
all_mongo_data = collection.find({})
for entry in all_mongo_data:
    for key, value in entry.items():
        if key == "embeddings":
            people_embeddings_arr.append(value)

THRESHOLD = 0.1 # thresh to qualify new entry

image = Image.open(image_path).convert("RGB")
image_trans = transforms(image).unsqueeze(0).to('cuda')

# Calculate image embeddings
with torch.no_grad():
    image_embeddings = model(image_trans).squeeze().cpu().numpy().astype(np.float32)
    image_embeddings /= np.linalg.norm(image_embeddings) # normalise

all_other_embeddings = np.stack(people_embeddings_arr)
pairwise_dist = cdist(image_embeddings.reshape(1, -1), all_other_embeddings.reshape(-1, 1000))
best_candidate_distance = np.min(pairwise_dist)

if best_candidate_distance < THRESHOLD:
    best_match_idx = np.argmin(pairwise_dist)
    print(f"Person found in DB at index {best_match_idx}")
else:
    print("Person not found in DB")
    people_embeddings_arr.append(image_embeddings)
    
client.close()

# walking, running, working, eating -> 0
# walking, running, eating, working -> 1
# ...

# Dataframe (csv, excel)
# 0,0,0,0 -> 0
# 0,0,1,0 -> 1
# 0,0,1,1 -> 2
# 1,1,1,1 -> 3

