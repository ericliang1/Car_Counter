from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

results = model.train(
    data="config.yaml", 
    epochs=30,         
    imgsz=640,  
)