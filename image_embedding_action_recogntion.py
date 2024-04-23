# action recognize

# from transformers import pipeline
# from PIL import Image

# # Define the model and processor name
# model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
# processor_name = "image-classification"

# # Load the pipeline with the specified model and processor
# pipe = pipeline(processor_name, model=model_name, device=0)  # Assuming 0 is the CUDA device index

# # Open and convert the image to RGB format
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11_pre.jpg").convert("RGB")

# # Process the image using the pipeline
# outputs = pipe(image)

# # Print the outputs
# print(outputs)
# print("testing")






# # image embedding
# import torch
# import numpy as np
# import torchvision
# from PIL import Image
# from scipy.spatial.distance import cdist

# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to('cuda')
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\test_images1\frame11_pre.jpg").convert("RGB")

# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize(
#         mean = [0.485, 0.456, 0.406], 
#         std = [0.229, 0.224, 0.225]
#     ),
# ])

# image_trans = transforms(image).unsqueeze(0).to('cuda')

# with torch.no_grad():
#     image_embeddings = model(image_trans).squeeze().cpu().numpy()

# print(image_embeddings)

# all_other_embeddings = np.stack([image_embeddings, image_embeddings, image_embeddings, image_embeddings])
# pairwise_dist = cdist(image_embeddings.reshape(1, -1), all_other_embeddings.reshape(-1, 1000))
# best_match_idx = np.argmin(pairwise_dist)
# print(pairwise_dist)



# import torch
# import numpy as np
# import torchvision
# from PIL import Image
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cdist

# # Load pre-trained ResNet-50 model
# model = torchvision.models.resnet50(pretrained=True).to('cuda')

# # Load and preprocess the image
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\test_images1\frame230_pre.jpg").convert("RGB")
# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize(
#         mean = [0.485, 0.456, 0.406], 
#         std = [0.229, 0.224, 0.225]
#     ),
# ])
# image_trans = transforms(image).unsqueeze(0).to('cuda')

# # Extract image embeddings
# with torch.no_grad():
#     image_embeddings = model(image_trans).squeeze().cpu().numpy()

# # Reduce the dimensionality of the embeddings using PCA
# pca = PCA(n_components=100)  # You can adjust the number of components
# image_embeddings_pca = pca.fit_transform(image_embeddings)

# print(image_embeddings_pca)





# action recognition along with date and action name

# from transformers import pipeline
# from PIL import Image
# from datetime import datetime
# import pymongo

# # Define the MongoDB connection
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["siddhant"]
# collection = db["face_data"]

# # Load the image classification pipeline
# processor_name = "image-classification"
# model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
# pipe = pipeline(processor_name, model=model_name, device=0)

# # Open and convert the image to RGB format
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11_pre.jpg").convert("RGB")

# # Process the image using the pipeline
# outputs = pipe(image)

# # Extract the action with the maximum confidence score
# max_action = max(outputs, key=lambda x: x['score'])

# # Get the current date and time
# timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # Store the face data and action data in the MongoDB collection
# face_data = {
#     "timestamp": timestamp,
#     "embedding_filename": "image_embeddings"  # Assuming this is the filename for face data
# }
# face_data_result = collection.insert_one(face_data)

# action_data = {
#     "timestamp": timestamp,
#     "action": max_action['label'],
#     "score": max_action['score'],
#     "face_data_id": face_data_result.inserted_id  # Store the _id of the face data document
# }
# collection.insert_one(action_data)

# # Print the timestamp, action, and its confidence score
# print(f"Timestamp: {timestamp}")
# print(f"Action: {max_action['label']}, Score: {max_action['score']}")
# print("Data stored in MongoDB")


# action and face only with time and date is not mentioned

# from transformers import pipeline
# from PIL import Image
# from datetime import datetime
# import pymongo

# # Define the MongoDB connection
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["siddhant1"]
# collection = db["face_data"]

# # Load the image classification pipeline
# processor_name = "image-classification"
# model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
# pipe = pipeline(processor_name, model=model_name, device=0)

# # Open and convert the image to RGB format
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\Images\frame11_pre.jpg").convert("RGB")

# # Process the image using the pipeline
# outputs = pipe(image)

# # Extract the action with the maximum confidence score
# max_action = max(outputs, key=lambda x: x['score'])

# # Get the current time
# time_now = datetime.now().strftime("%H:%M:%S")

# # Store the face data and action data in the MongoDB collection
# face_data = {
#     "timestamp": time_now,
#     "embedding_filename": "image_embeddings"  # Assuming this is the filename for face data
# }
# face_data_result = collection.insert_one(face_data)

# action_data = {
#     "timestamp": time_now,
#     "action": max_action['label'],
#     "score": max_action['score'],
#     "face_data_id": face_data_result.inserted_id  # Store the _id of the face data document
# }
# collection.insert_one(action_data)

# # Print the action, its confidence score, and the time
# print(f"Action: {max_action['label']}, Score: {max_action['score']}")
# print(f"Time: {time_now}")
# print("Data stored in MongoDB")



# import torch
# import numpy as np
# import torchvision
# from PIL import Image
# from sklearn.decomposition import PCA
# from pymongo import MongoClient

# # Load pre-trained ResNet-50 model
# model = torchvision.models.resnet50(pretrained=True).to('cuda')

# # Load and preprocess the image
# image = Image.open(r"C:\Users\siddh\Desktop\Final Year\test_images1\frame11_pre.jpg").convert("RGB")
# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], 
#         std=[0.229, 0.224, 0.225]
#     ),
# ])
# image_trans = transforms(image).unsqueeze(0).to('cuda')

# # Extract image embeddings
# with torch.no_grad():
#     image_embeddings = model(image_trans).squeeze().cpu().numpy()

# # Reduce the dimensionality of the embeddings using PCA
# pca = PCA(n_components=100)  # You can adjust the number of components
# image_embeddings_pca = pca.fit_transform(image_embeddings)

# # Connect to MongoDB
# client = MongoClient('localhost', 27017)
# db = client['tiki']
# collection = db['data']

# # Store the embeddings in MongoDB
# data = {'image_path': r"C:\Users\siddh\Desktop\Final Year\test_images1\frame230_pre.jpg", 'embeddings': image_embeddings_pca.tolist()}
# collection.insert_one(data)

# client.close()
