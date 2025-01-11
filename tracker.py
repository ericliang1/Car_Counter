import cv2
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

def car_counter_streamlit():
    st.set_page_config(page_title="Car and Truck Counter", layout="wide")

    # YOLO model setup
    model = YOLO("runs/detect/train10/weights/best.pt")
    tracker = DeepSort(max_age=20, nn_budget=70, max_iou_distance=0.8)

    # File uploader
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    actual_car_count = st.number_input("Enter the actual number of cars in the video:", min_value=0, step=1)

    if uploaded_video and st.button("Process Video"):
        # Save uploaded video
        input_path = f"input_{uploaded_video.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        # Load video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Uploaded video cannot be opened.")
            return

        # Retrieve video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video path
        output_path = f"output_{uploaded_video.name}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize counters
        unique_ids = set()

        # Set Parameters
        detection_threshold = 0.65
        iou_threshold = 0.8

        # Initialize progress bar
        progress_bar = st.progress(0)
        processed_frames = 0

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict Objects
            results = model.predict(frame, conf=detection_threshold, iou=iou_threshold)
            names = results[0].names

            # Filter for cars and trucks
            filtered_boxes = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                if cls_name in ["car", "truck"]:  # Treat both as cars
                    filtered_boxes.append(box)

            # Create Detections
            detections = [
                (
                    [
                        fb.xyxy[0][0],
                        fb.xyxy[0][1],
                        fb.xyxy[0][2] - fb.xyxy[0][0],
                        fb.xyxy[0][3] - fb.xyxy[0][1],
                    ],
                    1.0,
                    0
                )
                for fb in filtered_boxes
            ]

            # Update DeepSort tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Current IDs in frame
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

            # Write processed frame to output video
            out.write(frame)

            # Update progress bar
            processed_frames += 1
            progress = int((processed_frames / total_frames) * 100)
            progress_bar.progress(min(progress, 100))

        # Release resources
        cap.release()
        out.release()

        # Accuracy calculation
        if actual_car_count > 0:
            accuracy = (len(unique_ids) / actual_car_count) * 100
            accuracy_message = f"Accuracy: {accuracy:.2f}%"
        else:
            accuracy_message = "Actual car count not provided or is zero. Cannot calculate accuracy."

        # Inform user where the video is saved and show accuracy
        st.success("Video processing complete!")
        st.write(f"Processed video saved to `{output_path}` in the current folder.")
        st.write(f"Total Cars Detected: {len(unique_ids)}")
        st.write(accuracy_message)

        # Optionally cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    car_counter_streamlit()
