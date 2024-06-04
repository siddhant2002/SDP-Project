import os
import json
import random
import shutil
from ultralytics import YOLO

# Path to the directory containing images
image_directory = './training'

# Output directory for text files
output_directory = './labels'

# Unlabeled image directory for images that weren't detected by the model
unlabeled_directory = "./unlabeled"

# Model used to annotate
model = YOLO("./v8.pt")

os.makedirs(output_directory, exist_ok=True)


def createBackup(image_directory):
    randomNumber = random.randint(28, 9293)
    backup_directory = f"backup{randomNumber}"
    
    # Handle the situation where the backup directory already exists
    while os.path.exists(backup_directory):
        randomNumber = random.randint(28, 9293)
        backup_directory = f"backup{randomNumber}"
    
    try:
        shutil.copytree(image_directory, backup_directory)
        print(f"Backup created successfully: {backup_directory}")
    except Exception as e:
        print(f"Error creating backup: {e}")

# Creating backup
print("Creating backup!")
createBackup(os.path.join(image_directory))

# Loop through all files in the image directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        prediction = model.predict(os.path.join(image_directory, filename), save=True, stream=True)
        for r in prediction:  # Looping through the results
            if r:  # If result then execute the inside code 
                for box in r.boxes:
                    x, y, w, h = box.xywh[0].tolist() # Get the x, y, w, h coordinates.
                    

                    # Create the path for the text file in the output directory
                    txt_filename = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')

                    # Write the content to the text file
                    with open(txt_filename, 'w') as txt_file:
                        txt_file.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
            
                    print(f"Annotated {filename}")
            else:
                shutil.move(os.path.join(image_directory, filename), os.path.join(unlabeled_directory, filename))
                print(f"Failed to annotate {filename}. *No Detection by model*, skipping..")