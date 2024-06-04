from ultralytics import YOLO
import cv2
import os
import shutil

# Initialize YOLO model
model = YOLO('yolov8l.pt').to('cuda')

# Define folders
images_folder = "Images"
output_folder = 'test_images1'
bounding_data_folder = 'bounding_data'

# Create output and bounding data folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(bounding_data_folder, exist_ok=True)

# Paths for bounding box text files
txt_file_path = os.path.join(output_folder, 'bounding_boxes.txt')
bounding_data_txt_file_path = os.path.join(bounding_data_folder, 'bounding_boxes.txt')

# Initialize frame counter
frame_counter = 0

with open(txt_file_path, 'w') as txt_file:
    for img_file in os.listdir(images_folder):
        if img_file.endswith(".jpg"):
            frame_counter += 1
            if frame_counter % 1 != 0:  # Process every frame
                continue

            img_path = os.path.join(images_folder, img_file)
            results = model(img_path)

            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()

            img = cv2.imread(img_path)

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = conf
                detected_class = cls
                name = names[detected_class]
                if name == "person":
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (50, 200, 129), 2)
                    txt_file.write(f'{int(x1)} {int(y1)} {int(x2)} {int(y2)}\n')

            output_img_path = os.path.join(output_folder, f'{img_file[:-4]}_pre.jpg')
            cv2.imwrite(output_img_path, img)

# Move the bounding_boxes.txt file to the bounding_data folder
shutil.move(txt_file_path, bounding_data_txt_file_path)
