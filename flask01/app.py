# imports
import pickle
import random
from flask import Flask, render_template, Response, request
import cv2
import time
import os
import db
import mediapipe as mp
from string import Template

# import local files
import util as ut
import warnings

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

green = (0,204,0)
red = (0, 0, 255)

# create VideoCapture
vcap = cv2.VideoCapture(0)  # 0=camera

# mediapipe objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# string to frontend
global string
string = "hallo"
global image_index  # should be changed to a dictionary with strings and models, so the model gets changed too
image_index = 1

# load all images/poses
static_img_path = f'./static/img/'
images = os.listdir(static_img_path)
images = [x for x in images if x.endswith('.png')]
length = len(images)

# load model for downdog
model = pickle.load(open(f'models/model_downdog_svm.pkl', 'rb'))
# prevents spamming of UserWarnings while trying to fit model with missing landmarks
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------

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
    # fps = 15.0  # use different value to get slow-motion or fast-motion effect
    # fps = 30.0  # use different value to get slow-motion or fast-motion effect

    print('VCAP WIDTH :', width)
    print('VCAP HEIGHT:', height)
    print('VCAP FPS   :', fps)


def gen_frames():
    """ Displays webcam with landmarks generated by BlazePose/Mediapipe"""
    global detect_landmarks
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while vcap.isOpened():
            success, image = vcap.read()
            if not success:
                # Inform user.
                print("IGNORING EMPTY CAMERA FRAME.")
                break
            else:
                # Mark as not writeable to improve performance
                image.flags.writeable = False
                # Recolor image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Make detection
                results = pose.process(image)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    # Render Pose Detections
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # Extract landmarks from current frame
                    temp = []
                    landmarks = results.pose_landmarks.landmark
                    for i, j in zip(ut.used_landmarks, landmarks):
                        temp = temp + [j.x, j.y, j.z]
                    # Make predictions on the extracted landmarks
                    y = model.predict([temp])
                    if y == 1:
                        image = cv2.putText(image, "downdog", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 4)
                    if y == 0:
                        image = cv2.putText(image, "not downdog", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 4)

                # image = cv2.flip(image, 1)
                try:
                    # Create a buffer and return bytestream for webview
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
                except Exception as e:
                    pass
                if cv2.waitKey(10) & 0xFF == 27:
                    break


pcap = cv2.VideoCapture()  # non static poses


def gen_pose_frames():
    """TODO  Displays preprocessed poses in gif/video format"""

    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.6) as pose:
        while pcap.isOpened():
            success, image = pcap.read()
            if not success:
                break
            else:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


def gen_pose_img():
    img_path = f'./static/img/' + images[image_index]
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ret, buffer = cv2.imencode('.png', image)
    image = buffer.tobytes()
    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


# ---------------------------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pose_feed')
def pose_feed():
    return Response(gen_pose_img(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. In the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# define actions in runtime:
@app.route("/", methods=['POST'])
def tasks():
    global image_index
    if request.method == 'POST':
        if request.form.get('prev') == 'previous':
            image_index = (image_index - 1) % length
        elif request.form.get('nex') == 'next':
            image_index = (image_index + 1) % length
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=True)

# ---------------------------------------------------------------------

# on exit:
vcap.release()
pcap.release()

# Closes all the frames
cv2.destroyAllWindows()
