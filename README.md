# WORKSHOP-2--Object-detection-using-web-camera
# YOLOv8 Object Detection Using Laptop Camera
# Name : BUSHPIKA C
# register no  :212225230038
## Aim

To access the **laptop camera**, capture an image, and detect objects using **YOLOv8**.

## Requirements

* Anaconda
* Jupyter Notebook
* Laptop Camera

## Steps

1. Open **Jupyter Notebook** using Anaconda.
2. Access your **laptop camera** using OpenCV.
3. Wait for **5 seconds** and capture an image.
4. Display the captured image.
5. Apply **YOLOv8 object detection**.
6. Display the detected image with **bounding boxes and labels**.

## Algorithm

```text
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```

## Program
```
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
```
```
# Load a more accurate YOLO model
model = YOLO("yolov8m.pt")
```
```
# Open webcam
cap = cv2.VideoCapture(0)
```
```
# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,       
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

# Release webcam
cap.release()
cv2.destroyAllWindows()
```
## Output

<img width="898" height="623" alt="image" src="https://github.com/user-attachments/assets/e7f83235-5c09-4ecb-bafd-13ca3b21db09" />


## Result
The YOLOv8 model successfully loads and performs object detection using the webcam. The webcam captures video frames, and the model processes each frame to identify objects.
