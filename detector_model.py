from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

results = model.train(
    data="config.yaml", 
    epochs=10,         
    imgsz=640,    
    batch=32
)