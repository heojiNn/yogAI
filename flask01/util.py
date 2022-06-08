# TODO an app.py anbinden
import cv2
import mediapipe as mp
import numpy as np
import os


def load_pts(path):
    # loads data from path, return triencapsuled list like [DataEntry][Point][Coordinate]
    # returns None if Point is marked as not existent
    with open(path, mode="r", encoding='utf-8') as file:
        # data = np.ndarray()
        data = []
        for line in file:
            koords = line.split(";")
            pose = []
            for kord in koords:
                if kord == 'N/V':
                    pose.append([None, None, None])
                    continue
                if kord == '\n':
                    data.append(pose)
                    pose = []
                else:
                    # pose.append([float(k) for k in kord if k != ','])
                    pose.append([float(k) for k in kord.split(',')])
            # print(koords)
    return data
    # return True


def save_pts(path, data):
    # saves data into path, expects triencapsuled list or tuple like [DataEntry][Point][Coordinate]
    # saves NoneType point as "N/V;", skips NoneType entries
    with open(path, mode="w", encoding='utf-8') as file:
        for line in data:
            # print(str(line))
            if line is None:
                continue
            d = ""
            for point in line:
                if point is None:
                    file.write("N/V;")
                    continue
                d += str(point).replace("(", "").replace(")", "").replace(" ", "") + ";"
            file.write('%s\n' % d)
    return True


def entrypoint():
    data = ((None, (21,22,23), (31,32,33)), ((11,12,13), (21,22,23), (31,32,33)))
    #data = load_data("E:/YogAI/data/test/")
    path = 'tmp/data.txt'
    save_pts(path, data)
    print(load_pts(path))


def load_data(directory):
    # images inside the specified directory are loaded and converted to coordinates with generate_pts() and returned
    # as a list
    pose_list = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            pose_list.append(generate_pts(directory + filename))
    return pose_list


def generate_pts(path):
    # load img from path, process with blazepose and return list with coordinates of used_landmarks as (x, y, z) tuple
    # if no image or no landmarks are found under path None is returned
    mp_pose = mp.solutions.pose
    # the landmarks to be used
    used_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_PINKY,
        mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_INDEX,
        mp_pose.PoseLandmark.RIGHT_INDEX,
        mp_pose.PoseLandmark.LEFT_THUMB,
        mp_pose.PoseLandmark.RIGHT_THUMB,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,

    ]
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

        image = cv2.imread(path)
        if image is None:
            return None
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        # save all landmarks specified in used_landmarks
        landmarks = []
        for landmark in used_landmarks:
            landmarks.append((results.pose_landmarks.landmark[landmark].x,
                              results.pose_landmarks.landmark[landmark].y,
                              results.pose_landmarks.landmark[landmark].z))
        return landmarks


# functions can be called by app.py or /models/,,,.py directly
# if __name__ == '__main__':
#     entrypoint()
