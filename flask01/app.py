# imports
from flask import Flask, render_template, Response
import cv2
import time
import os
import db

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

# states
global running
running = True

# create VideoCapture
vcap = cv2.VideoCapture(0)  # 0=camera

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
    global running

    while running:
        success, frame = vcap.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        else:
            break
        # key = cv2.waitKey(1) & 0xFF  # get key (get only lower 8-bits to work with chars)
        # if key == KEY_Q or key == KEY_ESC:
        #     print("EXIT")
        #     running = False


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