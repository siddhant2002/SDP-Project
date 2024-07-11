from ultralytics import YOLO
import cv2
import os
import numpy as np
from deepface import DeepFace # type: ignore
from pymongo import MongoClient # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import uuid
from transformers import pipeline # type: ignore
from datetime import datetime
import torch
from PIL import Image

# Initialize YOLO model
model = YOLO('yolov8l.pt').to('cuda')

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_data']
collection = db['embedding']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the image classification pipeline for action recognition
processor_name = "image-classification"
model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
pipe = pipeline(processor_name, model_name, device=device)

# Global action map to store action IDs
action_map = {}
next_action_id = 1

# Function to check similarity with embeddings in the database
def check_similarity_with_database(new_embedding, similarity_threshold=0.65):
    try:
        max_similarity = -1  # Initialize to a very low value
        most_similar_document = None

        # Compare with embeddings in MongoDB collection
        for document in collection.find():
            # db_embedding = np.array(document["embedding"]).reshape(512)

            # # Calculate cosine similarity
            # similarity = cosine_similarity(new_embedding.reshape(512), db_embedding)
            db_embedding = np.array(document["embedding"])

            # Calculate cosine similarity
            similarity = cosine_similarity([new_embedding], [db_embedding])[0][0]

            # print(similarity)

            # Update max similarity and most similar document
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_document = document

        if max_similarity > similarity_threshold:
            print(f"Similar embedding found in database for person_id: {most_similar_document['person_id']} with similarity {max_similarity}")
            return most_similar_document  # Return the document if max similarity is above threshold

        return None  # Return None if no similar embedding is found above threshold

    except Exception as e:
        print(f"Error during similarity check: {e}")
        return None


# def cosine_similarity(embedding1, embedding2):
#     dot_product = np.dot(embedding1, embedding2)
#     norm1 = np.linalg.norm(embedding1)
#     norm2 = np.linalg.norm(embedding2)
#     return dot_product / (norm1 * norm2)

# Function to process images
def process_image(image_path, image_name):
    global next_action_id  # Use the global variable

    results = model(image_path)
    img = cv2.imread(image_path)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        confidence = conf
        detected_class = cls
        name = names[detected_class]
        if name == "person":
            # Crop the detected region
            cropped_img = img[y1:y2, x1:x2]

            # Convert the cropped image to PIL Image format and save temporarily
            pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            temp_img_path = "temp_cropped_img.jpg"
            pil_image.save(temp_img_path)

            # Perform embedding extraction
            try:
                embedding = DeepFace.represent(img_path=temp_img_path, model_name='Facenet512', enforce_detection=False)[0]['embedding']
                new_embedding = np.array(embedding)

                # Check if embedding already exists in database based on similarity
                similar_document = check_similarity_with_database(new_embedding)
                if similar_document:
                    print(f"Similar embedding found in database for person_id: {similar_document['person_id']}. Updating action object.")

                    # Perform action recognition
                    outputs = pipe(temp_img_path)
                    max_action = max(outputs, key=lambda x: x['score'])

                    # Assign an ID to the action name if it's not already assigned
                    if max_action['label'] not in action_map:
                        action_map[max_action['label']] = next_action_id
                        next_action_id += 1

                    action_id = action_map[max_action['label']]

                    # Get the current time
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Update the action data in the MongoDB collection
                    collection.update_one(
                        {"person_id": similar_document['person_id']},
                        {"$set": {f"actions.{time_now}": {"id": action_id, "name": max_action['label'], "confidence": max_action['score']}}}
                    )

                else:
                    # Perform action recognition
                    outputs = pipe(temp_img_path)
                    max_action = max(outputs, key=lambda x: x['score'])

                    # Assign an ID to the action name if it's not already assigned
                    if max_action['label'] not in action_map:
                        action_map[max_action['label']] = next_action_id
                        next_action_id += 1

                    action_id = action_map[max_action['label']]

                    # Get the current time
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Create document to insert into MongoDB
                    document = {
                        "person_id": uuid.uuid4().hex,
                        "image_name": image_name,
                        "embedding": new_embedding.tolist(),
                        "actions": {time_now: {"id": action_id, "name": max_action['label'], "confidence": max_action['score']}}
                    }

                    # Insert document into MongoDB collection
                    collection.insert_one(document)
                    print(f"Inserted embedding, image, and action for person_id: {document['person_id']} into MongoDB")

            except Exception as e:
                print(f"Error processing {image_name}: {e}")

# Main processing loop
image_directory = "Images"
for img_file in os.listdir(image_directory):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(image_directory, img_file)
        process_image(img_path, img_file)

print("Processing complete")

