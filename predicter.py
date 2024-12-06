from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  

results = model.predict(source='test_images', conf=0.1, save=True)  