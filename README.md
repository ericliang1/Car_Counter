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

1. Upload image files to the test_images folder
2. Run predicter.py in terminal using:
   
```bash
python predicter.py
```

3. Obtain predictions in runs/detect

---

To detect televisions using a video camera in real-time:

1. Run realtime_detector.py in terminal using:
   
```bash
streamlit run realtime_detector.py
```

---


