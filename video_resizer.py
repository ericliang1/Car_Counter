import cv2
import os

# Get the video file path from the user
input_file = input("Enter the path of the video file: ")

# Check if the file exists
if not os.path.isfile(input_file):
    print("Error: File not found.")
    exit()

# Desired dimensions (you can modify these or make them inputs as well)
new_width = 1080  # Replace with your desired width
new_height = 720 # Replace with your desired height

# Open the video file
cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original frame rate and codec
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Temporary output file
temp_file = "temp_resized_video.mp4"

# Set up VideoWriter
out = cv2.VideoWriter(temp_file, fourcc, fps, (new_width, new_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Write the frame to the output file
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Overwrite the original file
os.replace(temp_file, input_file)

print("Video resizing complete and the original file has been overwritten!")
