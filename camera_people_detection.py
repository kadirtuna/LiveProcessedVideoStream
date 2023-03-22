import time

import torch
import cv2
import numpy as np

class VideoPeopleDetection():
    def __init__(self):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.video_name = 'CrowdedAreaFootageVideo.mp4'

        # Read the video file
        self.cap = cv2.VideoCapture(self.video_name)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()

        # comparison =
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()

        # Convert frame to RGB and perform object detection with YOLOv5
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame, size=640)

        # Loop through each detected object and count the people
        num_people = 0
        for obj in results.xyxy[0]:
            if obj[-1] == 0:  # 0 is the class ID for 'person'
                num_people += 1
                # Draw bounding boxes around people
                xmin, ymin, xmax, ymax = map(int, obj[:4])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)


        # Draw the number of people on the frame and display it
        cv2.putText(frame, f'People: {num_people}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode(".jpg", frame)

        return jpeg.tobytes()
