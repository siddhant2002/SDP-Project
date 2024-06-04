from transformers import pipeline
import torch
import uuid
import torchvision
import numpy as np
from PIL import Image
from datetime import datetime
import pymongo
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
import torchvision.models as models

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["testing"]
collection = db["face_data"]

# # Load the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# Load the image classification pipeline
processor_name = "image-classification"
model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
pipe = pipeline(processor_name, model=model_name)

# Path to the Image1 folder
image_folder = r"C:\Users\siddh\Desktop\sdp project\SDP-Project\Images"

# Initialize variables for previous action
prev_action = None
prev_face_data_id = None
action_map = {}  # Map actions to numbers
action_ids = {}  # Dictionary to store action labels and their most recent IDs
next_action_id = 1  # Initialize the next action ID

# Preprocess the image
# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def add_data(data):
    try:
        _ = collection.insert_one(data)
        return "Success", 0
    except Exception as e:
        print(e)
        return "Failed", e

people_embeddings_arr = []
THRESHOLD = 0.1  # Threshold to qualify new entry

# Process each image in the Image1 folder
for filename in tqdm(os.listdir(image_folder)):
    if not filename.endswith(".png"):
        continue

    # Open and convert the image to RGB format
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")

    image_trans = transforms(image).unsqueeze(0).to('cuda')

    # Calculate image embeddings
    with torch.no_grad():
        image_embeddings = model(image_trans).squeeze().cpu().numpy().astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings)
        # print(image_embeddings)

    if not len(people_embeddings_arr):
        print("First add")
        people_embeddings_arr.append(image_embeddings)
        
        # Enter into DB
        data = {
            "person_id": uuid.uuid4().hex,
            "embeddings": image_embeddings.tolist()
        }
        
        status, message = add_data(data)
        if not message:
            print(message)

    else:
        # Check if any embedding is 'close enough' (determined by threshold) to this
        all_other_embeddings = np.stack(people_embeddings_arr)
        pairwise_dist = cdist(image_embeddings.reshape(1, -1), all_other_embeddings.reshape(-1, 1000))
        best_candidate_distance = np.min(pairwise_dist)

        if best_candidate_distance < THRESHOLD:
            best_match_idx = np.argmin(pairwise_dist)
            # print(f"Similar person found at index {best_match_idx}. Skipping...")
        else:
            print("New person. Adding")
            people_embeddings_arr.append(image_embeddings)
            
            # Enter into DB
            data = {
                "person_id": uuid.uuid4().hex,
                "embeddings": image_embeddings.tolist(),
                "actions": {}
            }
            status, message = add_data(data)
            if not message:
                print(message)

    # Process the image using the pipeline
    outputs = pipe(image)

    # Extract the action with the maximum confidence score
    max_action = max(outputs, key=lambda x: x['score'])

    # Get the current time
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Assuming you have a face_id for the person
    face_id = data["person_id"]

    # Assign an ID to the action name if it's not already assigned
    if max_action['label'] not in action_map:
        action_map[max_action['label']] = next_action_id
        next_action_id += 1

    # Update the action data in the MongoDB collection
    
    if max_action['score'] > 0.1:
        collection.update_one(
            {"person_id": face_id},
            {"$set": {f"actions.{time_now}": {"id": action_map[max_action['label']], "name": max_action['label'], "confidence": max_action['score']}}}
        )

    # Print the action, its confidence score, and the time
    print(f"Action: {max_action['label']}, ID: {action_map[max_action['label']]}, Confidence: {max_action['score']}, Timestamp: {time_now}")

client.close()

# {
#     "person_id_1": {
#         ts0: action,
#         ts1: action,
#         ts2: action, ...
#     },
#     "person_id_2": {
#         ts0: action,
#         ts1: action,
#         ts2: action, ...
#     },
#     "person_id_3": {
#         ts0: action,
#         ts1: action,
#         ts2: action, ...
#     },
# }