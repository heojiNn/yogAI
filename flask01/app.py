# std lib
import os
import time
import threading
import concurrent.futures as fut
from datetime import datetime

# flask
from flask import Flask, render_template, Response, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy  # package: 'Flask-SQLAlchemy'
from turbo_flask import Turbo  # package: 'Turbo-Flask'

# image and data processing
import cv2  # package: 'opencv-python'
import mediapipe as mp  # package: 'mediapipe' ;
import matplotlib.pyplot as plt
import numpy as np

# local
import routines.util as ut
import routines.WorkoutManager as wm
import routines.Ressources as rs
import evaluation.evaluation_util as ev
import gamification.avatar as av
import gamification.UserProfile as up

# ----------------------------------------------------------------------------------------------------------------------
# Initialize Flask-instance:
app = Flask(__name__.split('.')[0], instance_relative_config=True)
# Set up database: https://pythonbasics.org/flask-sqlalchemy/ ; [requirement: flask-sqlalchemy]
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scores.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# Set up for dynamic web content ; [requirement: Turbo-Flask]
turbo = Turbo(app)
global loc
global play  # option to manually start/pause a game later
play = False
# Create VideoCapture ; [requirement: opencv-python]
global vcap, pcap  # global: closed when on main-menu (fct main_menu), open while streaming (generator-functions)

# mediapipe objects ; [requirement: mediapipe]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# event_loop = asy.get_event_loop()
eval_landmarks = [[] for i in ut.used_landmarks]  # holds scans for each landmark
sum_eval_landmarks = []
sum_eval_landmarks2 = []
# data1 = [([], 'LEFT_WRIST'), ([], 'RIGHT_WRIST'), ([], 'LEFT_SHOULDER'), ([], 'RIGHT_SHOULDER')]
data1 = {'LEFT_WRIST': [], 'RIGHT_WRIST': [], 'LEFT_SHOULDER': [], 'RIGHT_SHOULDER': [], 'sum': [], }
video = []
poses = []
app.current_dist = [[1 for i in range(len(ut.used_landmarks))]]
app.current_dist2 = [[1 for i in range(len(ut.used_landmarks))]]
calc_frequency = 30  # loops to wait after each scan
eval_threshold = 10  # number of scans until evaluation
app.evaluate = False  # toggles evaluation-mode
health_update_freq = 30
app.gamefication = True
best_threshold = 0.2
medium_threshold = 0.35
feedback_frequency = 10  # how often feedback should be refreshed per second
global WoMan
WoMan = wm.WorkoutManager()
# access to workouts
workouts = rs.workouts.copy()  # workouts not read-only
wo_str = [wo.name for wo in workouts]  # display in dropdown
global wo_active
wo_active = ''
player1 = av.YogAvatar(av.Difficulty.hard)
player2 = av.YogAvatar(av.Difficulty.hard)
start_time = None
printed_highscore = False
# global buffer3dpoints
buffer3dpoints_p1 = np.zeros(shape=(12, 3))
buffer3dpoints_p2 = np.zeros(shape=(12, 3))
difficulties = ['easy', 'medium', 'hard']
# print(difficulties)
multiplayer_active = True


# ----------------------------------------------------------------------------------------------------------------------
# generator-functions:

# Functions:

def set_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)


def set_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)


def set_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)


def gen_points_in_background():  # like gen_frames but not implemented as generator
    global vcap
    global buffer3dpoints_p1, buffer3dpoints_p2, start_time
    vcap = cv2.VideoCapture(0)
    executor = fut.ThreadPoolExecutor()
    count = 0

    character_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1,
                      smooth_landmarks=True) as pose_left:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1,
                          smooth_landmarks=True) as pose_right:
            if not vcap.isOpened():
                print("ERROR INITIALIZING VIDEO CAPTURE")
                exit()
            else:
                print("OK INITIALIZING VIDEO CAPTURE")
                # get vcap property
                width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(vcap.get(cv2.CAP_PROP_FPS))
                ut.camFPS = fps
                drain_tick = None
                print('VCAP WIDTH :', width)
                print('VCAP HEIGHT:', height)
                print('VCAP FPS   :', fps)
            start_time = time.time()
            while vcap.isOpened():
                success, image = vcap.read()
                if not success:
                    print("IGNORING EMPTY CAMERA FRAME.")
                    break
                else:
                    if not multiplayer_active:
                        # Recolor image to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Make detection
                        results = pose_left.process(image)
                        if results.pose_landmarks is None:
                            continue
                        points3d = []
                        for lm in character_landmarks:
                            points3d.append((results.pose_landmarks.landmark[lm].x,  # pose_world_landmarks
                                             results.pose_landmarks.landmark[lm].y,
                                             results.pose_landmarks.landmark[lm].z))
                        for j, entry in enumerate(points3d):
                            if entry is None:
                                points3d[j] = buffer3dpoints_p1[j]
                        buffer3dpoints_p1 = points3d
                        count += 1
                        if count % calc_frequency == 0:
                            # video.append(image)
                            # poses.append(cv2.imread(WoMan.__img_path__))
                            fut_obj = executor.submit(get_dist, results)

                        old_tick = drain_tick
                        # print(f"old tick: {old_tick} new tick: {drain_tick}")
                        drain_tick = int((time.time() - start_time) * 10)
                        if old_tick != drain_tick and drain_tick % 10 == 0 and len(sum_eval_landmarks) \
                                and not WoMan.finished:
                            player1.drain(
                                sum_eval_landmarks[-1])  # TODO: pause drain when there are no points on screen

                        if WoMan.finished:
                            if not player1.saved:
                                active_user = get_active_user('multiplayer')[0]
                                create_user_workout('multiplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player1.score)
                                player1.saved = 1

                        cv2.waitKey(10)
                    else:
                        (h, w) = image.shape[:2]  # dims
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.flip(image, 1)
                        # split in sectors: left and right
                        left = image[:h, :w // 2]
                        right = image[:h, w // 2:]
                        # process landmarks in both regions:
                        results_left = pose_left.process(left)
                        results_right = pose_right.process(right)

                        old_tick = drain_tick
                        # print(f"old tick: {old_tick} new tick: {drain_tick}")
                        drain_tick = int((time.time() - start_time) * 10)
                        if old_tick != drain_tick and drain_tick % 10 == 0 and len(sum_eval_landmarks) \
                                and len(sum_eval_landmarks2) and not WoMan.finished:
                            print(f"drain_tick: {drain_tick}")
                            player1.drain(sum_eval_landmarks[-1], player2.combo)
                            # print("player2: ")
                            player2.drain(sum_eval_landmarks2[-1], player1.combo)

                        # --------------------------------------------------------------------------------------------------
                        # refresh bodypoints of Player 1 and Player 2
                        # Player1 on the left side of the camera
                        if results_left.pose_landmarks is None:
                            pass
                        else:
                            points3d = []
                            for lm in character_landmarks:
                                points3d.append((results_left.pose_landmarks.landmark[lm].x,  # pose_world_landmarks
                                                 results_left.pose_landmarks.landmark[lm].y,
                                                 results_left.pose_landmarks.landmark[lm].z))
                            for j, entry in enumerate(points3d):
                                if entry is None:
                                    points3d[j] = buffer3dpoints_p1[j]
                            buffer3dpoints_p1 = points3d

                        # Player2 on the right side of the camera
                        if results_right.pose_landmarks is None:
                            pass
                        else:
                            points3d = []
                            for lm in character_landmarks:
                                points3d.append((results_right.pose_landmarks.landmark[lm].x,  # pose_world_landmarks
                                                 results_right.pose_landmarks.landmark[lm].y,
                                                 results_right.pose_landmarks.landmark[lm].z))
                            for j, entry in enumerate(points3d):
                                if entry is None:
                                    points3d[j] = buffer3dpoints_p2[j]
                            buffer3dpoints_p2 = points3d

                        if WoMan.finished:
                            if not player1.saved:
                                active_user = get_active_user('multiplayer')[0]
                                create_user_workout('multiplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player1.score)
                                player1.saved = 1
                            if not player2.saved:
                                active_user = get_active_user('multiplayer')[1]
                                create_user_workout('multiplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player2.score)
                                player2.saved = 1
                        # --------------------------------------------------------------------------------------------------
                        # calc & draw:

                        if play:
                            #
                            ut.i += 1
                            if ut.i % calc_frequency == 0:
                                executor.submit(get_dist, results_left)
                                executor.submit(get_dist, results_right, mult=True)

                            # left.flags.writeable = True
                            # left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                            # ut.draw_pose_landmarks(left, results_left)
                            # if results_left.pose_landmarks is not None:
                            #     ut.highlight_denormalized_landmarks(left, results_left.pose_landmarks.landmark,
                            #                                         app.current_dist[0], [left.shape[0], left.shape[1]],
                            #                                         player1.thresh_300, player1.threshold)
                            # right.flags.writeable = True
                            # right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                            # ut.draw_pose_landmarks(right, results_right)
                            # if results_right.pose_landmarks is not None:
                            #     ut.highlight_denormalized_landmarks(right, results_right.pose_landmarks.landmark,
                            #                                         app.current_dist2[0], [right.shape[0], right.shape[1]],
                            #                                         player1.thresh_300, player1.threshold)

                            # drain health
                            if ut.i % health_update_freq == 0 and len(sum_eval_landmarks) > 0 and len(
                                    sum_eval_landmarks2) > 0:
                                # print("player1: ")
                                player1.drain(sum_eval_landmarks[-1])
                                # print("player2: ")
                                player2.drain(sum_eval_landmarks2[-1])

                    continue
            return


def gen_frames():
    """ Displays webcam with landmarks generated by BlazePose/Mediapipe"""
    global vcap, start_time
    vcap = cv2.VideoCapture(0)  # for webcam, 0=camera
    set_720p(vcap)      # set resolution of camera feed. Only set as high as your webcam allows
    executor = fut.ThreadPoolExecutor()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # check if video capturing has been initialized already
        if not vcap.isOpened():
            print("ERROR INITIALIZING VIDEO CAPTURE")
            exit()
        else:
            print("INITIALIZING VIDEO CAPTURE")
            # get vcap property
            drain_tick = None
            width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(vcap.get(cv2.CAP_PROP_FPS))
            print(f"width: {width}, heigth: {height}, fps: {fps}")
            ut.camFPS = fps
        start_time = time.time()  # TODO: handle reset action correctly
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
                image = cv2.flip(image, 1)
                # Flip image for better usability
                if ut.in_person_test: image = cv2.flip(image, 1)
                # Make detection
                results = pose.process(image)

                if play:
                    scale_factor = 5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_dead = cv2.FONT_HERSHEY_TRIPLEX
                    fontScale = 1
                    fontColor = (0, 0, 0)
                    fontColor_dead = (0, 0, 255)
                    thickness = 1  # smallest possible (another solution could be rendering a resized image with the data)
                    lineType = 1
                    pos = .05
                    workout_end = f'workout: {wo_active}'
                    score_end = f'achieved score: {player1.score}'
                    player1_max_combo = f'highest combo: x{int((player1.max_combo - 1) * 10)}'
                    if player1.dead:
                        WoMan.finished = True
                        image = np.zeros((image.shape[0], image.shape[1], 3))
                        image = cv2.putText(image,
                                            "YOU DIED",
                                            (int(image.shape[1] / 3), int(image.shape[0] / 2 * .9)),
                                            font_dead,
                                            fontScale * 4,
                                            fontColor_dead,
                                            thickness * 4,
                                            lineType)
                        image = cv2.putText(image,
                                            workout_end,
                                            (5, int(pos * image.shape[0])),
                                            font,
                                            fontScale,
                                            fontColor,
                                            thickness,
                                            lineType)
                        image = cv2.putText(image,
                                            score_end,
                                            (5,int(2 * pos * image.shape[0])),
                                            font,
                                            fontScale,
                                            fontColor,
                                            thickness,
                                            lineType)
                        image = cv2.putText(image,
                                            player1_max_combo,
                                            (5, int(3 * pos * image.shape[0])),
                                            font,
                                            fontScale,
                                            fontColor,
                                            thickness,
                                            lineType)
                    else:
                        # Add info-text to image:
                        timer = f'{int(time.time() - start_time)}s/{WoMan.workout.get_duration()}s'
                        workout_name = f'{wo_active}, {WoMan.workout.is_next_pose - 1}/{len(WoMan.workout.poses)}, '
                        score = f'score: {player1.score}'

                        # Render trainer in image (static):
                        if not WoMan.finished:
                            overlay = cv2.imread(WoMan.__img_path__)
                            image = cv2.putText(image,
                                                timer,
                                                (int(image.shape[1] * 1 / scale_factor) + 1, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                workout_name,
                                                (int(image.shape[1] * 1 / scale_factor) + 1, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                score,
                                                (int(image.shape[1] * 1 / scale_factor) + 1, int(4 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                        else:
                            image = cv2.putText(image,
                                                workout_end,
                                                (int(image.shape[1] * 1 / scale_factor) + 1, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                score_end,
                                                (int(image.shape[1] * 1 / scale_factor) + 1, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            overlay = cv2.imread('static/img/end.jpg')

                            # Save Highscore (active player):
                            if not player1.saved:
                                active_user = get_active_user('singleplayer')[0]
                                create_user_workout('singleplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player1.score)
                                player1.saved = 1

                        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        scale_width = int(image.shape[1] * 1 / scale_factor)
                        scale_height = int(image.shape[0] * 1 / scale_factor)
                        scale_dim = (scale_width, scale_height)
                        resized = cv2.resize(overlay, scale_dim, interpolation=cv2.INTER_AREA)
                        image[:height // scale_factor, :width // scale_factor] = resized
                        # image = cv2.addWeighted(image[:height//x, :width//x], 0.4, resized, 0.1, 0)
                        x = int(image.shape[1])
                        y = int(image.shape[0])
                        hp_width = 1 - 1 / scale_factor  # width in % of screen width
                        hp_height = .05
                        # top_left = int(x - hp_width * x), 0
                        # bottom_right = int(x * (1 - hp_width) + (player1.healthbar / 100) * x * hp_width), int(hp_height * y)
                        # bottom_origin = x, int(hp_height * y)

                        image = ut.put_healthbar(image, hp_width, hp_height, player1.healthbar, -1)

                        # image = ut.put_healthbar(image,
                        #                          top_left,
                        #                          bottom_right,
                        #                          -1,
                        #                          bottom_origin
                        #                          )

                        # TODO: Render trainer in image (transition):

                        # calculations and skeleton may start when user is ready
                        # Calc distance per landmark
                        ut.i += 1
                        if ut.i % calc_frequency == 0:
                            executor.submit(get_dist, results)
                            if app.evaluate:
                                video.append(image)
                                poses.append(cv2.imread(WoMan.__img_path__))  # TODO: append current pose

                        old_tick = drain_tick
                        # print(f"old tick: {old_tick} new tick: {drain_tick}")
                        drain_tick = int((time.time() - start_time) * 10)
                        if old_tick != drain_tick and drain_tick % 10 == 0 and len(sum_eval_landmarks) \
                                and not WoMan.finished:
                            player1.drain(sum_eval_landmarks[-1])  # TODO: pause drain when there are no points on screen
                        # # if len(eval_landmarks[0]) > eval_threshold and app.evaluate:
                        if WoMan.finished and app.evaluate:
                            app.evaluate = False
                            # plot_eval()
                            now = datetime.now().strftime("%m%d%Y_%H%M%S")
                            ev.generate_eval_vid(data1, video, poses, 10, f"evaluation/gifs/{now}")

                        # Draw the pose annotation on the image.
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # Render Pose Detections
                        ut.draw_pose_landmarks(image, results)
                        if results.pose_landmarks is not None:
                            ut.highlight_landmarks(image, results.pose_landmarks.landmark, app.current_dist[0],
                                                   player1.thresh_300, player1.threshold, [width, height])
                else:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                # Create a buffer and return bytestream for webview
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


def gen_frames_mult():
    """
        Displays webcam split in 2 sectors vertically.
        Within both sectors a landmark-skeleton can be drawn and read whenever a person can be detected.
    """
    global vcap, start_time
    vcap = cv2.VideoCapture(0)  # for webcam, 0=camera
    set_720p(vcap)
    executor = fut.ThreadPoolExecutor()
    # each person needs an individual mp_pose:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_left:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_right:
            if not vcap.isOpened():
                print("ERROR INITIALIZING VIDEO CAPTURE")
                exit()
            else:
                print("INITIALIZING VIDEO CAPTURE")
                # get vcap property
                width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(vcap.get(cv2.CAP_PROP_FPS))
                print(f"width: {width}, heigth: {height}, fps: {fps}")
                ut.camFPS = fps
                drain_tick = None
            start_time = time.time()
            while vcap.isOpened():
                success, image = vcap.read()
                if not success:
                    # Inform user.
                    print("IGNORING EMPTY CAMERA FRAME.")
                    break
                else:
                    (h, w) = image.shape[:2]  # dims
                    image.flags.writeable = False
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.flip(image, 1)
                    if ut.in_person_test: image = cv2.flip(image, 1)

                    if play:
                        # split in sectors: left and right
                        left = image[:h, :w // 2]
                        right = image[:h, w // 2:]
                        # process landmarks in both regions:
                        results_left = pose_left.process(left)
                        results_right = pose_right.process(right)

                        # --------------------------------------------------------------------------------------------------
                        # setup text:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_dead = font = cv2.FONT_HERSHEY_TRIPLEX
                        fontScale = 1
                        fontColor = (255, 255, 255)
                        fontColor_dead = (0, 0, 255)
                        thickness = 1  # smallest possible (another solution could be rendering a resized image with the data)
                        lineType = 1
                        pos = .05

                        workout_end = f'workout: {wo_active}'
                        difficulty = f'difficulty: {player1.difficulty}'
                        player1_endscore = f'achieved score: {player1.score}'
                        player2_endscore = f'achieved score: {player2.score}'
                        player1_max_combo = f'highest combo: x{int((player1.max_combo - 1) * 10)}'
                        player2_max_combo = f'highest combo: x{int((player2.max_combo - 1) * 10)}'

                        # check if player died
                        if player1.dead:
                            WoMan.finished = True
                            left = np.zeros((left.shape[0], left.shape[1], 3))
                            left = cv2.putText(left,
                                                "YOU DIED",
                                                (int(left.shape[1] / 3), int(left.shape[0] / 2 * .9)),
                                                font_dead,
                                                fontScale * 2,
                                                fontColor_dead,
                                                thickness * 2,
                                                lineType)
                        elif player2.dead:
                            WoMan.finished = True
                            right = np.zeros((right.shape[0], right.shape[1], 3))
                            right = cv2.putText(right,
                                                "YOU DIED",
                                                (int(right.shape[1] / 3), int(right.shape[0] / 2 * .9)),
                                                font_dead,
                                                fontScale * 2,
                                                fontColor_dead,
                                                thickness * 2,
                                                lineType)
                        else:
                            # --------------------------------------------------------------------------------------------------
                            # calc & draw:
                            ut.i += 1
                            if ut.i % calc_frequency == 0:
                                executor.submit(get_dist, results_left)
                                executor.submit(get_dist, results_right, mult=True)

                            left.flags.writeable = True
                            # left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                            ut.draw_pose_landmarks(left, results_left)
                            if results_left.pose_landmarks is not None:
                                ut.highlight_denormalized_landmarks(left, results_left.pose_landmarks.landmark,
                                                                    app.current_dist[0], [left.shape[0], left.shape[1]],
                                                                    player1.thresh_300, player1.threshold)
                            right.flags.writeable = True
                            # right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                            ut.draw_pose_landmarks(right, results_right)
                            if results_right.pose_landmarks is not None:
                                ut.highlight_denormalized_landmarks(right, results_right.pose_landmarks.landmark,
                                                                    app.current_dist2[0],
                                                                    [right.shape[0], right.shape[1]],
                                                                    player1.thresh_300, player1.threshold)

                        # drain health
                        old_tick = drain_tick
                        # print(f"old tick: {old_tick} new tick: {drain_tick}")
                        drain_tick = int((time.time() - start_time) * 10)
                        if old_tick != drain_tick and drain_tick % 10 == 0 and len(sum_eval_landmarks) \
                                and len(sum_eval_landmarks2) and not WoMan.finished:
                            print(f"drain_tick: {drain_tick}")
                            player1.drain(sum_eval_landmarks[-1], player2.combo)
                            # print("player2: ")
                            player2.drain(sum_eval_landmarks2[-1], player1.combo)

                        # --------------------------------------------------------------------------------------------------

                        # combine
                        image = np.concatenate((left, right), axis=1)
                        # draw line to visualize separator
                        image = cv2.line(image, (w // 2, 0), (w // 2, h), (255, 0, 180), 2)

                        # trainer placement
                        scale_factor = 5
                        scale_width = int(image.shape[1] * 1 / scale_factor)
                        scale_height = int(image.shape[0] * 1 / scale_factor)
                        scale_dim = (scale_width, scale_height)
                        trainer_x = int(width / 2 - scale_width / 2)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        fontColor = (0, 0, 0)
                        thickness = 1  # smallest possible (another solution could be rendering a resized image with the data)
                        lineType = 1
                        pos = .05
                        timer = f'{int(time.time() - start_time)}s/{WoMan.workout.get_duration()}s'
                        workout = f'{wo_active}, {WoMan.workout.is_next_pose - 1}/{len(WoMan.workout.poses)}, '
                        player1_score = f'score: {player1.score}'
                        player2_score = f'score: {player2.score}'

                        player1_combo = f'x{int((player1.combo - 1) * 10)}'
                        player2_combo = f'x{int((player2.combo - 1) * 10)}'

                        # Render trainer in image (static):
                        if not WoMan.finished:
                            overlay = cv2.imread(WoMan.__img_path__)

                            # text left
                            image = cv2.putText(image,
                                                workout,
                                                (5, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player1_score,
                                                (5, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player1_combo,
                                                (int((width - scale_width) / 2 + scale_width / 4),
                                                 int(scale_height * 1.2)),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)

                            # text right
                            image = cv2.putText(image,
                                                timer,
                                                (int((width + scale_width) / 2) + 5, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player2_score,
                                                (int((width + scale_width) / 2) + 5, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player2_combo,
                                                (int((width + scale_width) / 2 - scale_width / 2 + 5),
                                                 int(scale_height * 1.2)),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                        else:
                            # left
                            if player1.dead:
                                fontColor = (255, 255, 255)
                            image = cv2.putText(image,
                                                workout_end,
                                                (5, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player1_endscore,
                                                (5, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player1_max_combo,
                                                (5, int(4 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)

                            if player2.dead:
                                fontColor = (255, 255, 255)
                            else:
                                fontColor = (0, 0, 0)

                            # right
                            image = cv2.putText(image,
                                                difficulty,
                                                (int((width + scale_width) / 2) + 5, int(2 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player2_endscore,
                                                (int((width + scale_width) / 2) + 5, int(3 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            image = cv2.putText(image,
                                                player2_max_combo,
                                                (int((width + scale_width) / 2) + 5, int(4 * pos * image.shape[0])),
                                                font,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            overlay = cv2.imread('static/img/end.jpg')
                            if not player1.saved:
                                active_user = get_active_user('multiplayer')[0]
                                create_user_workout('multiplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player1.score)
                                player1.saved = 1
                            if not player2.saved:
                                active_user = get_active_user('multiplayer')[1]
                                create_user_workout('multiplayer', WoMan.workout.name)
                                update_user_workout(active_user.id, WoMan.workout.name, player2.score)
                                player2.saved = 1

                        resized = cv2.resize(overlay, scale_dim, interpolation=cv2.INTER_AREA)
                        # overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        image[:scale_height, trainer_x:trainer_x + scale_width] = resized

                        # put healthbars
                        hp_width = (1 - 1 / scale_factor) / 2
                        hp_height = .05
                        ut.put_healthbar_mult(image, hp_width, hp_height, player1.healthbar, player2.healthbar, -1)
                        # print(f"player1: {player1.healthbar}\nplayer2: {player2.healthbar}")
                    else:
                        image.flags.writeable = True
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # yield
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


def gen_2_frames():
    """
        Displays webcam split in 2 sectors vertically.
        Within both sectors a landmark-skeleton can be drawn and read whenever a person can be detected.
    """
    global vcap
    vcap = cv2.VideoCapture(0)  # for webcam, 0=camera
    set_720p(vcap)
    executor = fut.ThreadPoolExecutor()
    # each person needs an individual mp_pose:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_left:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_right:
            if not vcap.isOpened():
                exit()
            width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while vcap.isOpened():
                success, image = vcap.read()
                if success:
                    (h, w) = image.shape[:2]  # dims
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.flip(image, 1)
                    if ut.in_person_test: image = cv2.flip(image, 1)

                    if play:

                        # TODO:
                        # - Trainer left top corner
                        # - timer
                        # - score
                        # - healthbar
                        # copy/paste from singleplayer

                        x = 5

                        if not WoMan.finished:
                            overlay = cv2.imread(WoMan.__img_path__)
                            #
                        else:
                            overlay = cv2.imread('static/img/end.jpg')
                            #

                        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        scale_width = int(image.shape[1] * 1 / x)
                        scale_height = int(image.shape[0] * 1 / x)
                        scale_dim = (scale_width, scale_height)
                        resized = cv2.resize(overlay, scale_dim, interpolation=cv2.INTER_AREA)
                        image[:height // x, :width // x] = resized

                        # split in sectors: left and right
                        left = image[:h, :w // 2]
                        right = image[:h, w // 2:]
                        # process landmarks in both regions:
                        results_left = pose_left.process(left)
                        results_right = pose_right.process(right)
                        #
                        ut.i += 1
                        if ut.i % calc_frequency == 0:
                            executor.submit(get_dist, results_left)
                            executor.submit(get_dist, results_right, mult=True)

                        left.flags.writeable = True
                        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                        ut.draw_pose_landmarks(left, results_left)
                        if results_left.pose_landmarks is not None:
                            ut.highlight_denormalized_landmarks(left, results_left.pose_landmarks.landmark,
                                                                app.current_dist[0], [left.shape[0], left.shape[1]],
                                                                player1.thresh_300, player1.threshold)
                        right.flags.writeable = True
                        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                        ut.draw_pose_landmarks(right, results_right)
                        if results_right.pose_landmarks is not None:
                            ut.highlight_denormalized_landmarks(right, results_right.pose_landmarks.landmark,
                                                                app.current_dist2[0], [right.shape[0], right.shape[1]],
                                                                player1.thresh_300, player1.threshold)

                        # drain health
                        if ut.i % health_update_freq == 0 and len(sum_eval_landmarks) > 0 and len(
                                sum_eval_landmarks2) > 0:
                            # print("player1: ")
                            player1.drain(sum_eval_landmarks[-1])
                            # print("player2: ")
                            player2.drain(sum_eval_landmarks2[-1])

                        image = np.concatenate((left, right), axis=1)
                        # draw line to visualize separator
                        image = cv2.line(image, (w // 2, 0), (w // 2, h), (0, 0, 0), 1)
                    else:
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # yield
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


def gen_pose_img():
    """
        Current implementation alternates between poses and transitions depending on timers implemented
        in the WorkoutManager. All paths lead to a yield statement, delivering the image-bytestream to the frontend.
        Pose is static, to be dynamic it needs new cv2.VideoCapture (and loop to display frames).
        Transition is dynamic (video).
        Replay order: transition,pose,transition,...,pose,transition. transition is optional.
    """
    global WoMan, pcap, printed_highscore
    while play:
        if WoMan.workout.transitions is None:  # no transitions: just show images
            if not WoMan.finished:
                img_path = WoMan.__img_path__
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
            else:
                if not printed_highscore:
                    print(f"Workout finished! Score achieved: player1: {player1.score} player2: {player2.score}")
                    printed_highscore = True
                image = cv2.imread('static/img/end.jpg', cv2.IMREAD_COLOR)
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
        if WoMan.workout.transitions is not None:  # with transitions:
            if not WoMan.finished:
                pcap = cv2.VideoCapture(WoMan.workout.get_transition().path)
                # pcap.set(cv2.CAP_PROP_FPS, int(ut.camFPS))
                while pcap.isOpened() and WoMan.transition and not WoMan.kill:  # Show Transition
                    success, image = pcap.read()
                    if not success or WoMan.finished:
                        break
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'

                pcap.release()
                while not WoMan.transition and not WoMan.kill and not WoMan.finished:
                    img_path = WoMan.__img_path__
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
            else:
                image = cv2.imread('static/img/end.jpg', cv2.IMREAD_COLOR)
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


# utility-functions:

def calc_and_print_shit(inputs, player):  # function for euclidean score
    #  calc shit
    woman_pose = WoMan.workout.get_pose()
    if woman_pose.static or inputs[0] is None:
        inputs = inputs[1]
        inp_lm = []
        if not inputs.pose_landmarks:
            for lm in ut.used_landmarks:
                inp_lm.append(None)
        else:
            for lm in ut.used_landmarks:
                inp_lm.append((inputs.pose_landmarks.landmark[lm].x,
                               inputs.pose_landmarks.landmark[lm].y,
                               inputs.pose_landmarks.landmark[lm].z))

        comp = ut.compare_ptwise_static(inp_lm, WoMan.reference)
        app.current_dist = [np.array(comp)]
        print_statement = "Landmark | Input | Reference | Comparison\n"
        for lm, inp, gt, dist in zip(ut.used_landmarks, inp_lm, WoMan.reference, comp):
            print_statement += f"{lm} | {inp} | {gt} | {dist}\n"

        sum_landmarks = 0
        for i, entry in enumerate(comp):
            eval_landmarks[i].append(entry)
            sum_landmarks += entry
        score = sum_landmarks / len(comp)
        if player == 1:
            sum_eval_landmarks.append(score)
        else:
            sum_eval_landmarks2.append(score)

        if ut.verbose: print(print_statement + f"\n" + 50 * "-" + 10 * "\n")
    else:
        # make args
        pos = inputs[2]
        inputs = inputs[:2]

        # calc shit
        # print(f"current-gt-frame: {pos}")

        # pic points (old) <=> LM Vectors (old) <=?=> WoMan.workout.Pose.Data[app.VideoCounter (-1)]
        # Load processed reference frames
        pic_points_old = woman_pose.select_spec_frame(pos=int(pos - 1))
        pic_points = woman_pose.select_spec_frame(pos=pos)

        proc_gt = [pic_points_old, pic_points]

        print("IM HERE!")
        # Processing input frames
        proc_inputs = []
        for inp in inputs:
            inp_lm = []
            if not inp.pose_landmarks:
                for lm in ut.used_landmarks:
                    inp_lm.append(None)
            else:
                for lm in ut.used_landmarks:
                    inp_lm.append((inp.pose_landmarks.landmark[lm].x,
                                   inp.pose_landmarks.landmark[lm].y,
                                   inp.pose_landmarks.landmark[lm].z))
            proc_inputs.append(inp_lm)

        static_inp = proc_inputs[1]

        # calc scores
        print(f"processing of pos: {pos} started!")
        rel_speed_diff, abs_speed_diff, unit_dist, angles = ut.compare_ptwise_dynamic(proc_inputs, proc_gt,
                                                                                      woman_pose.FPS)
        ptwise_dist = ut.compare_ptwise_static(inp=static_inp, gt=pic_points)

        sum_landmarks = 0
        for i, entry in enumerate(ptwise_dist):
            eval_landmarks[i].append(entry)
            sum_landmarks += entry
        score = sum_landmarks / len(ptwise_dist)
        if player == 1:
            sum_eval_landmarks.append(score)
        else:
            sum_eval_landmarks2.append(score)

        # Refresh current scores located in app.current_dist if new one is found
        app.current_dist = np.array([ptwise_dist, rel_speed_diff, abs_speed_diff, unit_dist, angles])
        for i, it in enumerate([ptwise_dist, rel_speed_diff, abs_speed_diff, unit_dist, angles]):
            for j, item in enumerate(it):
                if item is None:
                    continue
                app.current_dist[i, j] = item


def plot_eval():
    # fig, ax = plt.subplots(1, len(ut.used_landmarks))

    X = [range(len(lm)) for lm in eval_landmarks]
    labels = [str(lm) for lm in ut.used_landmarks]

    for i in range(len(ut.used_landmarks)):
        plt.figure(i + 1)
        plt.title(labels[i])
        plt.plot(X[i], eval_landmarks[i])
        # plt.set_title
        # plot.set_title(labels[i])
        j = i

    for k, img in enumerate(video):
        plt.figure(k + j + 2)
        plt.imshow(img)
        plt.title(str(k))

    plt.show()


def get_workout_by_name(name):
    w = [w for w in workouts if w.name == name]
    if len(workouts) == 0:
        Flask.abort(404)
    return w[0]


def get_dist(lms, mult=False):  # short version of 'calc_and_print_shit' also suited for multiplayer
    global data
    woman_pose = WoMan.workout.get_pose()
    if woman_pose.static or lms[0] is None:
        lms = lms[1]
        inp_lm = []
        if not lms.pose_landmarks:
            for lm in ut.used_landmarks:
                inp_lm.append(None)
        else:
            for lm in ut.used_landmarks:
                inp_lm.append((lms.pose_landmarks.landmark[lm].x,
                               lms.pose_landmarks.landmark[lm].y,
                               lms.pose_landmarks.landmark[lm].z))

        comp = ut.compare_ptwise_static(inp_lm, WoMan.reference)
        if mult:
            app.current_dist2 = [np.array(comp)]
        else:
            app.current_dist = [np.array(comp)]

        sum_landmarks = 0
        for i, entry in enumerate(comp):
            eval_landmarks[i].append(entry)
            sum_landmarks += entry
        score = sum_landmarks / len(comp)

        if app.evaluate:
            data1["LEFT_WRIST"].append(comp[4])
            data1["RIGHT_WRIST"].append(comp[5])
            data1["LEFT_SHOULDER"].append(comp[0])
            data1["RIGHT_SHOULDER"].append(comp[1])
            data1["sum"].append(score)

        if mult:
            sum_eval_landmarks2.append(score)
        else:
            sum_eval_landmarks.append(score)


# ----------------------------------------------------------------------------------------------------------------------
# Routing/Frontend:


def update_load():
    with app.app_context(), app.test_request_context():
        while True:
            if player1.dead:
                return redirect(url_for('gameover'))
            time.sleep(1 / feedback_frequency)
            if turbo.can_push():
                try:
                    turbo.push(turbo.replace(render_template('infopanels/gamepanel.html'), 'health'))
                    turbo.push(turbo.replace(render_template('infopanels/gamepanel_multiplayer.html'), 'health_mult'))
                    turbo.push(turbo.replace(render_template('html-pages/game3d.html'), 'databuffer'))
                    # print("Values updated!")
                except Exception as e:
                    # print(f"Exception occured:\n{str(e)}")
                    pass


@app.context_processor
def inject_load():
    labels = [str(lm).split(sep='.')[1] for lm in ut.used_landmarks]
    # print(f"character.healthbar: {character.healthbar}")
    # skel_labels = [f"__dot {lbl}" for lbl in labels]
    # zipped_data = zip(skel_labels, app.current_dist[0])
    pose = WoMan.workout.get_pose()
    posename = pose.path.upper().split('.')[0]
    if multiplayer_active:
        users = get_active_user('multiplayer')
        player1_name = users[0].name.upper()
        player2_name = users[1].name.upper()
    else:
        player1_name = get_active_user('singleplayer')[0].name.upper()
        player2_name = "NOT_ACTIVE"
    player1_data = buffer3dpoints_p1.copy()
    player2_data = buffer3dpoints_p2.copy()
    player1_combo = int((player1.combo - 1) * 10)
    player2_combo = int((player2.combo - 1) * 10)

    trainer_data = np.array(WoMan.reference).copy()
    return_dict = {  # 'data': app.current_dist,
        # 'head': labels, 'best': best_threshold, 'medium': medium_threshold,
        'health1': player1.healthbar, 'health2': player2.healthbar,
        'score1': player1.score, 'score2': player2.score,
        'combo1': player1_combo, 'combo2': player2_combo,
        'PoseName': posename,

        'left_shoulder_pts_p1': list(player1_data[0]),
        'left_elbow_pts_p1': list(player1_data[1]), 'left_wrist_pts_p1': list(player1_data[2]),
        'left_hip_pts_p1': list(player1_data[3]),
        'left_knee_pts_p1': list(player1_data[4]), 'left_ankle_pts_p1': list(player1_data[5]),
        'right_shoulder_pts_p1': list(player1_data[6]),
        'right_elbow_pts_p1': list(player1_data[7]), 'right_wrist_pts_p1': list(player1_data[8]),
        'right_hip_pts_p1': list(player1_data[9]),
        'right_knee_pts_p1': list(player1_data[10]), 'right_ankle_pts_p1': list(player1_data[11]),

        'left_shoulder_pts_p2': list(player2_data[0]),
        'left_elbow_pts_p2': list(player2_data[1]), 'left_wrist_pts_p2': list(player2_data[2]),
        'left_hip_pts_p2': list(player2_data[3]),
        'left_knee_pts_p2': list(player2_data[4]), 'left_ankle_pts_p2': list(player2_data[5]),
        'right_shoulder_pts_p2': list(player2_data[6]),
        'right_elbow_pts_p2': list(player2_data[7]), 'right_wrist_pts_p2': list(player2_data[8]),
        'right_hip_pts_p2': list(player2_data[9]),
        'right_knee_pts_p2': list(player2_data[10]), 'right_ankle_pts_p2': list(player2_data[11]),

        'left_shoulder_pts_trainer': list(trainer_data[0]),
        'left_elbow_pts_trainer': list(trainer_data[2]), 'left_wrist_pts_trainer': list(trainer_data[4]),
        'left_hip_pts_trainer': list(trainer_data[6]),
        'left_knee_pts_trainer': list(trainer_data[8]), 'left_ankle_pts_trainer': list(trainer_data[10]),
        'right_shoulder_pts_trainer': list(trainer_data[1]),
        'right_elbow_pts_trainer': list(trainer_data[3]), 'right_wrist_pts_trainer': list(trainer_data[5]),
        'right_hip_pts_trainer': list(trainer_data[7]),
        'right_knee_pts_trainer': list(trainer_data[9]), 'right_ankle_pts_trainer': list(trainer_data[11]),

        'name_player1': player1_name, 'name_player2': player2_name,

        'map_name': 'SwedishRoyalCastle', 'multiplayer': multiplayer_active
    }
    return return_dict


@app.before_first_request
def before_first_request():
    global wo_active
    WoMan.change_workout(rs.SUN_SALUTATION)  # set a default Workout to be active on startup
    wo_active = WoMan.workout.name  # active workout routine-string (needed for dropdown)


gen_points_in_background_thread = threading.Thread(target=gen_points_in_background)
update_load_thread = threading.Thread(target=update_load)


@app.route('/game3d', methods=['POST', 'GET'])
def game3d():
    global loc, gen_points_in_background_thread, update_load_thread
    loc = 'game3d'
    print("Game3d entered")
    if not gen_points_in_background_thread.is_alive():
        gen_points_in_background_thread = threading.Thread(target=gen_points_in_background)
        gen_points_in_background_thread.start()
    print("gen_points_in_background started")
    if not update_load_thread.is_alive():
        update_load_thread = threading.Thread(target=update_load)
        update_load_thread.start()
    if play:
        threading.Thread(target=WoMan.start).start()
    print("All Threads Started!")
    return render_template('html-pages/game3dcontainer.html', workouts=wo_str, active=wo_active, mp=False, play=play,
                           loc=loc, difs=difficulties)


@app.route('/')
def menu():
    player1.reset()
    player2.reset()
    return render_template('html-pages/menu.html')


@app.route('/gameover')
def gameover():
    return render_template('infopanels/gameover.html')


@app.route('/pose_feed')
def pose_feed():
    return Response(gen_pose_img(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. In the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/multiplayer_feed')
def multiplayer_feed():
    return Response(gen_frames_mult(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/main_menu', methods=['GET', 'POST'])
@app.route('/main_menu/<int:score_page>', methods=['GET', 'POST'])  # used to distinguish from scores.html
def main_menu(score_page=0):
    player1.reset()
    player2.reset()
    global vcap, WoMan
    if request.method == 'POST':
        if score_page == 0:
            workout = WoMan.workout.reset()
            WoMan.kill = True
            try:
                vcap.release()  # if you click on menu without the camera initialization finished NameError is thrown
            except NameError:
                pass
            WoMan = wm.WorkoutManager()
            WoMan.change_workout(workout)
    return redirect(url_for('menu'))


@app.route('/singleplayer', methods=['GET', 'POST'])
def singleplayer():
    player1.reset()
    player2.reset()
    global loc, printed_highscore, start_time
    start_time = time.time()
    loc = 'singleplayer'
    printed_highscore = False
    if play:
        threading.Thread(target=WoMan.start).start()
        threading.Thread(target=update_load).start()
        # threading.Thread(target=update_load).start()
    return render_template('html-pages/game_page.html', workouts=wo_str, active=wo_active, mp=False, play=play, difs=difficulties)


@app.route('/multiplayer', methods=['GET', 'POST'])
def multiplayer():
    player1.reset()
    player2.reset()
    global loc, printed_highscore, start_time
    start_time = time.time()
    loc = 'multiplayer'
    printed_highscore = False
    threading.Thread(target=WoMan.start).start()
    threading.Thread(target=update_load).start()
    return render_template('html-pages/game_page.html', workouts=wo_str, active=wo_active, mp=True, play=play, difs=difficulties)


@app.route('/scores', methods=['GET', 'POST'])
def scores():
    player1.reset()
    player2.reset()
    global loc
    loc = 'scores'
    users = db.session.query(User.id, User.name, User.mode, User.active).all()
    wkts = db.session.query(Workout.name, Workout.highest).all()
    user_wkts = db.session.query(UserWorkout.id, UserWorkout.name, UserWorkout.user_highest).all()
    return render_template('html-pages/scores.html', Users=users, Workouts=wkts, UserWorkouts=user_wkts)


@app.route('/pick', methods=['GET', 'POST'])
def pick():
    player1.reset()
    player2.reset()
    global wo_active, WoMan, start_time
    start_time = time.time()
    if request.method == 'POST':
        selection = request.form.get('workout')
        WoMan.kill = True
        workout = get_workout_by_name(selection).reset()
        WoMan = wm.WorkoutManager()
        WoMan.change_workout(workout)
        wo_active = selection
    return redirect(url_for(loc))


@app.route('/pick_difficulty', methods=['GET', 'POST'])
def pick_difficulty():
    player1.reset()
    player2.reset()
    global wo_active, WoMan, start_time
    start_time = time.time()
    if request.method == 'POST':
        selection = request.form.get('workout')
        av.set_difficulty(av.Difficulty[selection])
        WoMan.kill = True
        workout = get_workout_by_name(selection).reset()
        WoMan = wm.WorkoutManager()
        WoMan.change_workout(workout)
        wo_active = selection
    return redirect(url_for(loc))


@app.route('/restart', methods=['GET', 'POST'])
def restart():
    global WoMan, start_time
    start_time = time.time()
    player1.reset()
    player2.reset()
    if request.method == 'POST':
        WoMan.kill = True
        workout = WoMan.workout.reset()
        WoMan = wm.WorkoutManager()
        WoMan.change_workout(workout)
    return redirect(url_for(loc))


@app.route('/un_play', methods=['POST'])
def un_play():
    global play, WoMan, start_time
    start_time = time.time()
    player1.reset()
    player2.reset()
    if request.method == 'POST':
        play = not play  # swap play state
        WoMan.kill = True
        workout = WoMan.workout.reset()
        WoMan = wm.WorkoutManager()
        WoMan.change_workout(workout)
    return redirect(url_for(loc))


@app.route('/switch_multiplayer', methods=['GET', 'POST'])
def switch_multiplayer():
    global multiplayer_active, start_time
    start_time = time.time()
    if request.method == 'POST':
        multiplayer_active = not multiplayer_active
    return redirect(url_for(loc))


@app.route('/user_add/<mode>', methods=['GET', 'POST'])
def user_add(mode):
    if request.method == 'POST':
        try:
            selection = request.form.get('add')
            create_user(name=selection, mode=mode)
        except ValueError:
            print('invalid name')
    return redirect(url_for(loc))


@app.route('/user_delete', methods=['GET', 'POST'])
def user_delete():
    if request.method == 'POST':
        try:
            selection = int(request.form.get('del'))
            print(selection)
            delete_user(selection)
        except Exception as e:
            print(f'invalid id: {str(e)}')
    return redirect(url_for(loc))


@app.route('/set_active/<mode>', methods=['GET', 'POST'])
def set_active(mode):
    if request.method == 'POST':
        try:
            if mode == 'singleplayer':
                selection = int(request.form.get('set'))
                set_active_user(selection, mode)
            elif mode == 'multiplayer':
                selection = request.form.get('set')
                try:
                    x = selection.split(",")
                    p1 = int(x[0])
                    p2 = int(x[1])
                except Exception as e:
                    print(f'conversion of ids failed : {str(e)}')
                set_active_user(p1, mode, p2)
            else:
                print('invalid mode')
        except ValueError:
            print('invalid id')
    return redirect(url_for(loc))


# ----------------------------------------------------------------------------------------------------------------------
# flask-sqlalchemy Models(=Tables):
# https://flask-sqlalchemy.palletsprojects.com/en/2.x/


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=True)
    mode = db.Column(db.String(200), nullable=False)  # -> two different user necessary depending on the gamemode
    active = db.Column(db.Boolean, nullable=False)

    def __repr__(self):
        return '<User-id %r>' % self.id


class Workout(db.Model):
    name = db.Column(db.String(200), primary_key=True)  # name as unique id
    highest = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return '<Workout %r>' % self.name


class UserWorkout(db.Model):
    id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='CASCADE', name="fk_uw_id"),
        primary_key=True
    )
    name = db.Column(
        db.String(200),
        db.ForeignKey('workout.name', ondelete='CASCADE', name="fk_uw_name"),
        primary_key=True
    )
    user_highest = db.Column(db.Integer, nullable=True)

    user = db.relationship('User', backref='user_scores')
    workout = db.relationship('Workout', backref='workout_scores')

    def __repr__(self):
        return '<User-High-score %r>' % self.user_highest


# ----------------------------------------------------------------------------------------------------------------------
# CRUD <User>:

def create_user(name, mode, activity=False):
    db.session.add(User(name=name, mode=mode, active=activity))
    db.session.commit()


def read_user(mode=None):  # returns active user(s)
    mode_users = User.query.filter_by(mode=mode).all()
    if mode_users is None:
        print('no active user/s found!')
        return
    if mode == 'singleplayer':  # only one user can be active in singleplayer
        for user in mode_users:
            if user.active:
                return up.UserProfile(user.id, user.name, user.mode, user.active)
    elif mode == 'multiplayer':
        users = []
        for user in mode_users:
            if user.active:
                users.append(up.UserProfile(user.id, user.name, user.mode, user.active))
        return users
    else:
        print('invalid gamemode specified')
        return


def update_user(uid, name=None, mode=None):
    try:
        user = User.query.filter_by(id=uid).first()
        if name is not None:
            user.name = name
        if mode is not None:
            user.mode = mode
        if name is not None or mode is not None:
            db.session.commit()
    except Exception as e:
        print(f'updating User {uid} failed: {e}')
        pass


def delete_user(uid):
    try:
        User.query.filter(User.id == uid).delete()
        db.session.commit()
    except Exception as e:
        print(f'deleting User {uid} failed: {e}')
        pass


def get_active_user(mode):
    mode_users = User.query.filter_by(mode=mode).all()
    return [i for i in mode_users if i.active]


def set_active_user(uid, mode, secondary_id=1):
    mode_users = User.query.filter_by(mode=mode).all()
    old_active = [i for i in mode_users if i.active and i.mode == mode]
    # set inactive/active:
    if mode == 'singleplayer':  # only one user can be active in singleplayer
        user = User.query.filter_by(id=uid).first()
        user.active = True
        for user in old_active:
            if not user.id == uid:
                user.active = False
        db.session.commit()
    elif mode == 'multiplayer' and secondary_id is not None:  # two users can be active in multiplayer
        user1 = User.query.filter_by(id=uid).first()
        user2 = User.query.filter_by(id=secondary_id).first()
        user1.active = True
        user2.active = True
        for user in old_active:
            if not user.id in [secondary_id, uid]:
                user.active = False
        new_active = [i for i in mode_users if i.active]
        if len(new_active) != 2:
            print('invalid amount of active players')
            return
        db.session.commit()
    elif secondary_id is None:
        print('secondary ID not specified')
    else:
        print('invalid gamemode specified')
        return


def check_active_user():
    return NotImplementedError

# CRUD <Workout>:


def create_workout():
    # creating here currently does nothing because of 'initialise_db'
    return NotImplementedError


def read_workout(name):
    # return highscore
    w = Workout.query.filter_by(name=name).first()
    if w is None:
        print(f'workout {name} not found!')
        return
    return w.highest


def update_workout():
    # update to highscore
    return NotImplementedError


def delete_workout():
    # deleting here currently does nothing because of 'initialise_db'
    return NotImplementedError


# CRUD <UserWorkout>:

def create_user_workout(mode, workout, mult=False):
    """ Creates the initial entry when it doesn't exist yet. """
    user = get_active_user(mode)
    if read_user_workout(user[0].id, workout) is not None:
        print(f'UserWorkout entry for {user[0].id, workout} already exists.')
        return
    else:
        print(f'Creating new UserWorkout entry for {user[0].id, workout}.')
        for u in user:  # max 2 iters
            db.session.add(UserWorkout(id=u.id, name=workout, user_highest=0))
        db.session.commit()


def read_user_workout(uid, workout):
    uw = UserWorkout.query.filter_by(id=uid, name=workout).first()
    if uw is not None:
        return uw
    else:
        return None


def update_user_workout(uid, workout, new_score):
    uw_old = read_user_workout(uid, workout)
    if uw_old is None:
        print('updating user failed (entry doesn\'t exist)')
        return
    if uw_old.user_highest is not None:
        if new_score > uw_old.user_highest:
            uw_old.user_highest = new_score
    else:
        uw_old.user_highest = new_score
    db.session.commit()


def delete_user_workout(uid, workout):
    try:
        User.query.filter(UserWorkout.id == uid, UserWorkout.name == workout).delete()
        db.session.commit()
    except Exception as e:
        print(f'deleting UserWorkout {uid, workout} failed: {e}')
        pass


# Util:

def initialise_db():
    db.create_all()  # create all tables/models
    entries = db.session.query(Workout.name).all()  # get all existing Workout-entries by column 'name'
    ex = [item for t in entries for item in t]  # convert list of tuples to list of strings
    for wo in wo_str:  # compare database workouts (by name) with code-workouts
        if wo not in ex:  # add missing or do nothing (to avoid primary key collisions)
            # score defaulted to 100 * cnt(pics)
            db.session.add(Workout(name=wo, highest=100 * len(get_workout_by_name(wo).poses)))
    default = User.query.filter_by(name='default_user').first()
    if default is None:  # None is returned if entry is missing ...
        new_user = User(name='default_user', mode='singleplayer', active=True)
        db.session.add(new_user)
    default1 = User.query.filter_by(name='player1').first()
    default2 = User.query.filter_by(name='player2').first()
    if default1 is None:
        create_user(name='player1', mode='multiplayer', activity=True)
    if default2 is None:
        create_user(name='player2', mode='multiplayer', activity=True)

    db.session.commit()


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    initialise_db()
    app.run(host='localhost', debug=True)

# ----------------------------------------------------------------------------------------------------------------------
