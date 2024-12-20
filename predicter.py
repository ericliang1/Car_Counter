from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/best.pt")  

results = model.predict(source='test_images', conf=0.4, save=True)  