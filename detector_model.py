import torch
from ultralytics import YOLO

print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.version)
print(torch.cuda.is_available())

model = YOLO("yolov8s.pt")

results = model.train(
    data="config.yaml", 
    epochs=100,         
    imgsz=640,  
    batch=4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
)