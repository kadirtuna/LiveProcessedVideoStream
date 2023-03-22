from flask import Flask, render_template, Response
#from camera_face_detection import VideoCamera
from camera_people_detection import VideoPeopleDetection
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoPeopleDetection()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")