import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

#Configuration
model = YOLO("runs/detect/train2/weights/best.pt")
tracker = DeepSort(max_age=20, nn_budget=70, max_iou_distance=0.8)
video_path = "cars.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Retrieve video settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#Create Output Video
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set Parameters
detection_threshold = 0.65  
iou_threshold = 0.8  
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict Objects
    results = model.predict(frame, conf=detection_threshold, iou=iou_threshold)

    # Create Detections
    detections = [
        ([box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2] - box.xyxy[0][0], box.xyxy[0][3] - box.xyxy[0][1]], 1.0, 0)
        for box in results[0].boxes
    ]

    # Update Deepsort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    current_ids = set() 
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bounding boxes
        label = f"Car ID: {track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        unique_ids.add(track_id)
        current_ids.add(track_id)

    # Display total and current car counts
    total_cars = len(unique_ids)
    current_cars = len(current_ids)
    cv2.putText(
        frame,
        f"Total Cars: {total_cars}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Cars in Frame: {current_cars}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Create output video
    out.write(frame)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total Cars Detected: {len(unique_ids)}")
