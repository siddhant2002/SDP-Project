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
import pickle
from twilio.rest import Client # type: ignore
from prophet import Prophet

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["siddhant1"]
collection = db["face_data"]

# Initialize an empty dictionary to map actions to their IDs
action_map = {}

cursor = collection.find({}, {"_id": 0, "actions": 1})
for doc in cursor:
    for action_data in doc["actions"].values():
        action_label = action_data["name"]
        action_id = action_data["id"]
        if action_label not in action_map:
            action_map[action_label] = action_id


print(action_map)
# Load the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# Path to the Image1 folder
image_path = r"C:/Users/siddh/Desktop/Final Year/test_images1/frame1510_pre.jpg"

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
    # Define the model and processor name
    model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
    processor_name = "image-classification"

    # Load the pipeline with the specified model and processor
    pipe = pipeline(processor_name, model=model_name, device=0)  # Assuming 0 is the CUDA device index

    # Open and convert the image to RGB format

    # Process the image using the pipeline
    outputs = pipe(image)

    # Get the action with the highest confidence score
    max_action = max(outputs, key=lambda x: x['score'])

    # Print the action and its confidence score
    print(f"Action: {max_action['label']}, Confidence: {max_action['score']}")

    # Load the trained model
    with open('prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # model = Prophet()
    # model = model.load_model('prophet_model.pkl')

    # Get the current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Make a prediction for the current time
    future = model.make_future_dataframe(periods=1, freq='T', include_history=False)
    forecast = model.predict(future)

    # Get the forecasted action for the current time
    forecasted_action = forecast['yhat'][0]
    yhat_lower = forecast['yhat_lower'][0]
    yhat_upper = forecast['yhat_upper'][0]

    # Compare predicted action with the action in the new image
    threshold = 0.8  # Set your threshold here
    action_id = action_map.get(max_action['label'], "Action not found")
    print(forecasted_action)
    print(action_id)
    print(yhat_lower)
    print(yhat_upper)
    # is_anomaly = abs(forecasted_action - action_id) > threshold
    is_anomaly = action_id <= yhat_upper and action_id >= yhat_lower

    if action_id == "Action not found" or not is_anomaly:
        print("Anomaly detected!")
        # account_sid = 'AC37a8b368ea50b05576c12d79d2b94b14'
        # auth_token = 'e4142ffdd38e8097aa7e4c2bff7c93a1'

        # # Your Twilio phone number and the number you want to send the SMS to
        # twilio_phone_number = '+12055095953'
        # recipient_phone_number = '+917205873440'

        # # Create a Twilio client
        # client = Client(account_sid, auth_token)

        # # Send the SMS
        # client.messages.create(
        #     to=recipient_phone_number,
        #     from_=twilio_phone_number,
        #     body='Anomaly detected! Deep in the computer.'
        # )

        # print('SMS alert sent successfully.')
    else:
        print("No anomaly detected.")
else:
    print("Person not found in DB")
    people_embeddings_arr.append(image_embeddings)

# client.close()
