# imports
from flask import Flask, render_template, Response
import cv2
import time
import os
import db
import mediapipe as mp

# ---------------------------------------------------------------------

# initialization of global Flask-class instance:
# ref: https://flask.palletsprojects.com/en/2.1.x/tutorial/factory/

app = Flask(__name__.split('.')[0], instance_relative_config=True)
app.config.from_mapping(
    # a default secret that should be overridden by instance config
    SECRET_KEY="dev",
    # store the database in the instance folder
    DATABASE=os.path.join(app.instance_path, "sqlite"),
)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

# register the database commands
db.init_app(app)

# ---------------------------------------------------------------------

# premature implementation of the webcam-display:

KEY_Q = ord('q')  # quit
KEY_ESC = 27  # quit

# create VideoCapture
vcap = cv2.VideoCapture(0)  # 0=camera

# mediapipe objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# check if video capturing has been initialized already
if not vcap.isOpened():
    print("ERROR INITIALIZING VIDEO CAPTURE")
    exit()
else:
    print("OK INITIALIZING VIDEO CAPTURE")
    # get vcap property
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(vcap.get(cv2.CAP_PROP_FPS))
    # fps = 15.0  # use different value to get slowmotion or fastmotion effect
    # fps = 30.0  # use different value to get slowmotion or fastmotion effect

    print('VCAP width :', width)
    print('VCAP height:', height)
    print('VCAP fps   :', fps)


def gen_frames():  # generate frame by frame from camera
    """ Heart of the image generation/display process. Gets called by http/flask-frontend.
     """
    # applying BlazePose
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while vcap.isOpened():
            success, image = vcap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display. (doesn't work when embedded in flask/webpage)
            # cv2.flip(image, 1)

            # Create a buffer and return bytestream for webview
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'

            # abort with key 27 doesn't work on flask/webpage - should be a route/http-action?
            if cv2.waitKey(5) & 0xFF == 27:
                break


@app.route('/video_feed')
def video_feed():
    # Video streaming route. In the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------------------------------------------------

# startup routes and web-functionality:

@app.route('/')
def index():
    return render_template('index.html')


# define actions in runtime:
# @app.route("/task")
# def task():
#     time.sleep(2)
#     return '<span>Done</span>'


if __name__ == '__main__':
    app.run(host='localhost', debug=True)

# ---------------------------------------------------------------------

# on exit:

vcap.release()
# result.release() # in case recording happens

# Closes all the frames
cv2.destroyAllWindows()