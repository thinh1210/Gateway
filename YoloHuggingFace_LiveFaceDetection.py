import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import PIL.Image
from IPython.display import display, clear_output
import os
from dotenv import load_dotenv

# load environment variables from a .env file to hide sensible information
load_dotenv=(r"huggingface_access.env")

HF_TOKEN=os.getenv("HF_TOKEN")

# function to draw rectangles around detected face objects
def draw_rectangles(frame, detections):
    # iterate over each detection
    for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = bbox.astype(int)  # get the bounding box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw the rectangle
        class_name = detections.data['class_name'][class_id]  # get the class name
        label = f"{class_name}: {confidence:.2f}"  # create the label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # draw the label
    return frame

# download the YOLOv8 model from hugging_face 
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load the model
model = YOLO(model_path)

# start video capture with the webcam
cap = cv2.VideoCapture(0)

while True:
    # capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # perform inference with the model
    output = model(frame)
    results = Detections.from_ultralytics(output[0])

    # draw rectangles around detected faces
    frame_with_rectangles = draw_rectangles(frame, results)

    # show the frame with circled objects
    frame_rgb = cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(frame_rgb)
    cv2.imshow('Frame', frame_with_rectangles)

    # check if the 'q' key was pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()