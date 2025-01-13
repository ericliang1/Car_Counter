  # Car Counter

**Car Counter** is a car tracking model created using yolov8, the coco dataset, and deepsort

---

## Features

- Counting total number of cars in a video

---

## Requirements

Before using the project, ensure you have the following:

- python
- ultralytics
- deepsort
- opencv
- pytorch

Install required libraries with:

```bash
   pip install -r requirements.txt
   ```

---

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ericliang1/Car_Counter.git
   cd Car_Counter
   ```
---

## Usage

### Run the Project

To count cars in a video:

1. Run the streamlit app in terminal using:

```bash
streamlit run tracker.py
```

2. Upload the video to the file uploader and enter the number of actual cars in the video in the box

3. Obtain annotated video and accuracy metric after progress bar has loaded
   
---

## Demonstration

![](https://github.com/ericliang1/Car_Counter/blob/main/output_cars.gif)

---


