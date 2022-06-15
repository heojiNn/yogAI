import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = [
    mpPose.PoseLandmark.NOSE,
    mpPose.PoseLandmark.LEFT_EYE,
    mpPose.PoseLandmark.RIGHT_EYE,
    mpPose.PoseLandmark.LEFT_EAR,
    mpPose.PoseLandmark.RIGHT_EAR,
    mpPose.PoseLandmark.LEFT_SHOULDER,
    mpPose.PoseLandmark.RIGHT_SHOULDER,
    mpPose.PoseLandmark.LEFT_ELBOW,
    mpPose.PoseLandmark.RIGHT_ELBOW,
    mpPose.PoseLandmark.LEFT_WRIST,
    mpPose.PoseLandmark.RIGHT_WRIST,
    mpPose.PoseLandmark.LEFT_PINKY,
    mpPose.PoseLandmark.RIGHT_PINKY,
    mpPose.PoseLandmark.LEFT_INDEX,
    mpPose.PoseLandmark.RIGHT_INDEX,
    mpPose.PoseLandmark.LEFT_THUMB,
    mpPose.PoseLandmark.RIGHT_THUMB,
    mpPose.PoseLandmark.LEFT_HIP,
    mpPose.PoseLandmark.RIGHT_HIP,
    mpPose.PoseLandmark.LEFT_KNEE,
    mpPose.PoseLandmark.RIGHT_KNEE,
    mpPose.PoseLandmark.LEFT_ANKLE,
    mpPose.PoseLandmark.RIGHT_ANKLE,
    mpPose.PoseLandmark.LEFT_HEEL,
    mpPose.PoseLandmark.RIGHT_HEEL,
    mpPose.PoseLandmark.LEFT_FOOT_INDEX,
    mpPose.PoseLandmark.RIGHT_FOOT_INDEX,
]
#
#


def extract_landmarks(path_dir, save_dir, name, label, landmarks_used=points):
    # pathdir is where pictures are read from
    # name is suffix of new .csv - file
    # label is added to each pose for classification, can be None
    # landmarks_used is a list of the Pose-landmarks that should be extracted
    data_tmp = []
    for p in landmarks_used:
        x = str(p)[13:]
        data_tmp.append(x + "_x")
        data_tmp.append(x + "_y")
        data_tmp.append(x + "_z")
    if label is not None:
        data_tmp.append('label')
    data_tmp = pd.DataFrame(columns=data_tmp)  # Empty dataset

    count = 0
    for img in os.listdir(path_dir):
        temp = []
        img = cv2.imread(path_dir + "/" + img)
        if img is None:
            continue

        imagewidth, imageheight = img.shape[:2]
        blackie = np.zeros(img.shape)  # Blank image
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
            mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # draw landmarks on blackie
            landmarks = results.pose_landmarks.landmark
            for i, j in zip(landmarks_used, landmarks):
                temp = temp + [j.x, j.y, j.z]
            if label is not None:
                temp.append(label)
            data_tmp.loc[count] = temp
            count += 1
        # visualization of the extraction process
        cv2.imshow("Image", img)
        cv2.imshow("blackie", blackie)
        cv2.waitKey(100)

    if save_dir is not None:
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        data_tmp.to_csv(f"{save_dir}/dataset_{name}.csv")  # save the data as a csv file
    else:
        data_tmp.to_csv(f"dataset_{name}.csv")  # save the data as a csv file
#
#


def extract_landmarks_v2(path_dir, save_dir, name, label, landmarks_used=points):
    # difference: read all pictures of all directories lower or equal to <path_dir>
    data_tmp = []
    for p in landmarks_used:
        x = str(p)[13:]
        data_tmp.append(x + "_x")
        data_tmp.append(x + "_y")
        data_tmp.append(x + "_z")
    if label is not None:
        data_tmp.append('label')
    data_tmp = pd.DataFrame(columns=data_tmp)  # Empty dataset

    count = 0
    dirnames = [x[1] for x in os.walk(path_dir)][0]
    for k in range(len(dirnames)):
        for img in os.listdir(path_dir + "/" + dirnames[k]):
            temp = []
            img = cv2.imread(path_dir + "/" + dirnames[k] + "/" + img)
            if img is None:
                continue
            blackie = np.zeros(img.shape)
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
                mpDraw.draw_landmarks(blackie, results.pose_landmarks,
                                      mpPose.POSE_CONNECTIONS)  # draw landmarks on blackie
                landmarks = results.pose_landmarks.landmark
                for i, j in zip(landmarks_used, landmarks):
                    temp = temp + [j.x, j.y, j.z]
                if label is not None:
                    temp.append(label)
                    # temp.append(dirnames[k])  # adding pos_name to dataframe
                data_tmp.loc[count] = temp
                count += 1
            # visualization of the extraction process
            cv2.imshow("Image", img)
            cv2.imshow("blackie", blackie)
            cv2.waitKey(100)

    if save_dir is not None:
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        data_tmp.to_csv(f"{save_dir}/dataset_{name}.csv")  # save the data as a csv file
    else:
        data_tmp.to_csv(f"dataset_{name}.csv")  # save the data as a csv file
#
#


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          decimals=2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=decimals)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_label_distribution(labels, classes=['No Diabetis', 'Diabetis']):
    """
    This function plots the distribution of the given labels
    """
    label_count = [int(list(labels).count(x)) for x in range(len(classes))]
    y_pos = np.arange(len(classes))

    plt.bar(y_pos, label_count, align='center')
    plt.xticks(y_pos, classes)
    plt.ylabel('Occurences')
    plt.title('Label')
    plt.show()


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
                elif kord == '0' or kord == '1':
                    pose.append(int(kord))
                elif kord == '':
                    continue
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


# load images from dir and generate landmarks:


def load_data(directory, label):
    # images inside the specified directory are loaded and converted to coordinates with generate_pts() and returned
    # as a list
    pose_list = []
    for file in os.listdir(directory):
        print("reading img")
        filename = os.fsdecode(file)
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            pose_list.append(generate_pts(directory + filename, label))
    return pose_list


def generate_pts(path, label):
    # load img from path, process with blazepose and return list with coordinates of used_landmarks as (x, y, z) tuple
    # if no image or no landmarks are found under path None is returned
    mp_pose = mp.solutions.pose
    # the landmarks to be used - 26
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

        # append a label for each image-skeleton
        if label is not None:
            landmarks.append(label)
        return landmarks


# if __name__ == '__main__':
#     entrypoint()




