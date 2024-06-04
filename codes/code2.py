# from ultralytics import YOLO
# import cv2
# import os

# model = YOLO('yolov8l.pt').to('cuda')

# img_path = "Images/frame1510.jpg"

# results = model(img_path)

# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# names = results[0].names
# confidences = results[0].boxes.conf.tolist()

# output_folder = 'Images'
# os.makedirs(output_folder, exist_ok=True)

# img = cv2.imread(img_path)

# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     confidence = conf
#     detected_class = cls
#     name = names[detected_class]
#     line = f"{x1},{y1},{x2},{y2},{detected_class},{confidence}\n"
#     with open(os.path.join(output_folder, 'bounding_boxes.txt'), 'a') as f:
#         if name == "person":
#             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (50, 200, 129), 2)
#             # cv2.putText(img, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
# cv2.imwrite(os.path.join(output_folder, f'frame1510_pre.jpg'), img)



from ultralytics import YOLO
import cv2
import os


model = YOLO('yolov8l.pt').to('cuda')

images_folder = "Images"
output_folder = 'test_images1'
os.makedirs(output_folder, exist_ok=True)
frame_counter = 0  # Initialize frame counter

for img_file in os.listdir(images_folder):
    if img_file.endswith(".jpg"):
        frame_counter += 1  # Increment frame counter
        if frame_counter % 1 != 0:  # Skip frames that are not multiples of 20
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
                # cv2.putText(img, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imwrite(os.path.join(output_folder, f'{img_file[:-4]}_pre.jpg'), img)