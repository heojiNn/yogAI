import cv2
import mediapipe as mp
import numpy as np
import os
# for other utils
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

mp_pose = mp.solutions.pose
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
    data = ((None, (21, 22, 23), (31, 32, 33)), ((11, 12, 13), (21, 22, 23), (31, 32, 33)))
    # data = load_data("E:/YogAI/data/test/")
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
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      enable_segmentation=True, min_detection_confidence=0.5) as pose:
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


# ---------------------------------------------------------------------
# OTHER UTIL: https://github.com/pereldegla/yoga_assistant/blob/master/utils.py
# ---------------------------------------------------------------------


# Build the dataset using landmarks detection and save it as csv
def build_dataset(path, dataset_type):
    # path: ROOT PATH TO DATASET
    # dataset_type: type of dataset
    data = []
    for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")
    data.append("target")  # name of the position
    data = pd.DataFrame(columns=data)  # Empty dataset
    count = 0

    dirnames = [x[1] for x in os.walk(path)][0]
    # walking through the whole training dataset
    for k in range(len(dirnames)):
        for img in os.listdir(path + "/" + dirnames[k]):
            temp = []
            img = cv2.imread(path + "/" + dirnames[k] + "/" +img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for i, j in zip(points, landmarks):

                    temp = temp + [j.x, j.y, j.z, j.visibility]

                temp.append(dirnames[k]) #adding pos_name to dataframe

                data.loc[count] = temp
                count +=1
    data.to_csv(dataset_type+".csv") # save the data_train as a csv file | viewing on ExcelReader might suck


# Predict the name of the poses in the image
def predict(img, model, show=False):
    temp = []
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
            temp = temp + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp])

        if show:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            cv2.putText(img, str(y[0]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
            cv2.imshow("image", img)
            cv2.waitKey(0)


def predict_video(model, video="0", show=False):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        temp = []
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]
            y = model.predict([temp])
            name = str(y[0])
            if show:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(img, (40, 40), (40+w, 60), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.imshow("Video", img)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


# Use this to evaluate any dataset you've built
def evaluate(data_test, model, show=False):
    target = data_test.loc[:, "target"]  # list of labels
    target = target.values.tolist()
    predictions = []
    for i in range(len(data_test)):
        tmp = data_test.iloc[i, 0:len(data_test.columns) - 1]
        tmp = tmp.values.tolist()
        predictions.append(model.predict([tmp])[0])
    if show:
        print(confusion_matrix(predictions, target), '\n')
        print(classification_report(predictions, target))
    return predictions


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
points = mpPose.PoseLandmark  # Landmarks
# build_dataset("DATASET/TRAIN", "train")
# build_dataset("DATASET/TEST", "test")


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def create_hull(label, landmarks_used=used_landmarks):
    data_tmp = []
    for p in landmarks_used:
        x = str(p)[13:]
        data_tmp.append(x + "_x")
        data_tmp.append(x + "_y")
        data_tmp.append(x + "_z")
    if label is not None:
        data_tmp.append('label')
    return pd.DataFrame(columns=data_tmp)  # Empty dataset


