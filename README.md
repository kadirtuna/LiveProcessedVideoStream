# People Counter Server Broadcast
 The Python Flask server publishes the processed video which detects and counts the people frame by frame with Neural Network Algorithm Yolo Model.
 
 The Live Processed Video Stream Project is targeted that the image processed video in Python's streaming to a WEB Page. As the target point, Flask Framework and Yolov5s model has been choosed to perform the project. Before all on the image processing stage, any video which has some people to be detected is taken by the script. With yolov5s model, the frames of a video or webcam is scanned to find some people on it. If there is any person in the frame, the script wraps up every people with the boundin box seperately. Also, it counts the people and print out on the frame. Furthermore, all of these processed image is streamed in runtime by designed with Flask Framework. To try it in your local network, you should follow the source code of the project in Github.
 
<p align="center">
 <img src="https://github.com/kadirtuna/LiveProcessedVideoStream/blob/main/Images/LiveProcessedVideoStream.jpg">
</img>
</p>
