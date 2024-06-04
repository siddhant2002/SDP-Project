from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official detection model
model = YOLO("yolov8n-seg.pt")  # load an official segmentation model
# model = YOLO("path/to/best.pt")  # load a custom model
 
results = model.track(source="video2.mp4", show=True, tracker="botsort.yaml")