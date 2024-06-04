import os
from pymongo import MongoClient
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8l.pt').to('cuda')

path='videos/video5.mp4'
frame = cv2.imread(path)
vidObj = cv2.VideoCapture(path) 
success = 1
while success: 
		# vidObj object calls read 
		# function extract frames 
    success, image = vidObj.read() 

    # Saves the frames with frame-count 
    if success:
        img_path = "Images/frame11.jpg"
        img = cv2.imread(img_path)

        # Perform inference on a video
        results = model('videos/video3.mp4')

        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['siddhant1']
        collection = db['bounding_boxes']

        # Define the path to the output folder
        output_folder = 'Images'

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save bounding boxes to a text file and store in MongoDB
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            confidence = conf
            detected_class = cls
            name = names[int(cls)]
            line = f"{x1},{y1},{x2},{y2},{detected_class},{confidence}\n"
            with open(os.path.join(output_folder, 'bounding_boxes.txt'), 'a') as f:
                if name.__eq__("person"):
                    f.write(line)
                    collection.insert_one({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class': name,
                        'confidence': confidence
                    })
                    # Draw bounding box on the frame
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (50, 200, 129), 2)
                    
                    # Save the frame with bounding box
                    cv2.imwrite(os.path.join(output_folder, f'frame_{x1}_{y1}_{x2}_{y2}.jpg'), frame)

        # f.write(line)
        # collection.insert_one({
        #     'x1': x1,
        #     'y1': y1,
        #     'x2': x2,
        #     'y2': y2,
        #     'class': name,
        #     'confidence': confidence
        # })

    # Read the frame