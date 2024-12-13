import time
import cv2
import torch
import streamlit as st
from ultralytics import YOLO

#create real-time detection app

def inference():

    #configure settings

    st.set_page_config(page_title="Real-Time Television Detector", layout="wide", initial_sidebar_state="auto")

    webcam = 0  

    model = YOLO("runs/detect/train4/weights/best.pt")  
    class_names = list(model.names.values())

    selected_classes = ["television"]
    selected_index = [class_names.index(option) for option in selected_classes]

    #set minimum detection thresholds

    conf = 0.25  
    iou = 0.45

    #create video boxes

    col1, col2 = st.columns(2)
    col1.write("**Original**")
    col2.write("**Annotated**")
    original = col1.empty()
    annotated = col2.empty()

    fps_display = st.sidebar.empty()
    st.sidebar.title("Real-Time Detection")

    #test if start button pressed

    if st.sidebar.button("Start"):
        videocapture = cv2.VideoCapture(webcam) 

        if not videocapture.isOpened():
            st.error("Webcam can not be opened")

        stop_button = st.button("Stop") 

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Webcam failed to read frames")
                break

            prev_time = time.time()  

            #find detections

            results = model(frame, conf=conf, iou=iou, classes=selected_index)
            annotated_frame = results[0].plot()

            #record fps

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)

            #display original video alongside annotated video

            original.image(frame, channels="BGR")
            annotated.image(annotated_frame, channels="BGR")

            #test if stop button pressed

            if stop_button:
                videocapture.release()  
                torch.cuda.empty_cache() 
                st.stop() 

            fps_display.metric("FPS", f"{fps:.2f}")

        videocapture.release()

    torch.cuda.empty_cache()
    print(class_names)

if __name__ == "__main__":
    inference()