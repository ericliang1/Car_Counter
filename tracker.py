import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("runs/detect/train2/weights/best.pt") 

tracker = DeepSort(max_age=50, nn_budget=70)


video_path = "cars.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output_with_bounding_boxes.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


detection_threshold = 0.5
car_class_id = 0  
unique_car_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        if class_id == car_class_id and confidence > detection_threshold:
            width = x2 - x1
            height = y2 - y1
            detections.append(([x1, y1, width, height], float(confidence), car_class_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb()) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        unique_car_ids.add(track_id)

    out.write(frame)

cap.release()
out.release()

print(f"Total Cars Detected: {len(unique_car_ids)}")
print(f"Processed video saved as: {output_path}")
