import datetime
import os
import time

import torch
import cv2
import numpy as np

class VideoPeopleDetection():
    time_reference = datetime.datetime.now()
    counter_frame = 0
    processed_fps = 0
    def __init__(self):
        # Load YOLOv5 model
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        modelName = "best_person_29.05.2023-yolov5s-2.pt"
        self.model = self.load_model(modelName)
        self.classes = self.model.names
        self.video_name = 'CrowdedAreaFootageVideo.mp4'
        # self.video_name = 'For_Validation6.mp4'

        # Read the video file
        self.cap = cv2.VideoCapture(self.video_name)

    def __del__(self):
        self.cap.release()

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load((os.getcwd()) + "\\ultralytics_yolov5_master", 'custom', source='local', path=model_name, force_reload=True)

        return model
    def class_to_label(self, x):
        return self.classes[int(x)]
    def get_frame(self):
        ret, frame = self.cap.read()

        # comparison =
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()

        # Convert frame to RGB and perform object detection with YOLOv5
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model(frame, size=640)

        # Loop through each detected object and count the people
        num_people = 0
        bgr = (0, 255, 0)

        #To get the processed FPS
        #VideoPeopleDetection.time_reference = datetime.datetime.now()

        time_now = datetime.datetime.now()
        time_diff = (time_now - VideoPeopleDetection.time_reference).seconds

        if time_diff >= 1:
            VideoPeopleDetection.time_reference = datetime.datetime.now()
            VideoPeopleDetection.processed_fps = VideoPeopleDetection.counter_frame
            VideoPeopleDetection.counter_frame = 0
        else:
            VideoPeopleDetection.counter_frame += 1

        for obj in results.xyxy[0]:
            if obj[-1] == 0:  # 0 is the class ID for 'person'

                # Draw bounding boxes around people
                xmin, ymin, xmax, ymax = map(int, obj[:4])
                accuracy = obj[4]
                if (accuracy > 0.5):
                    num_people += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(frame, f" {round(float(accuracy), 2)}", (xmin, ymin),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        # Draw the number of people on the frame and display it
        cv2.putText(frame, f'FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'People: {num_people}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Processed FPS: {VideoPeopleDetection.processed_fps}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode(".jpg", frame)

        return jpeg.tobytes()
