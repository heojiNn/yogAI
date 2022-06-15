import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image as PImage
import util as ut
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from util import plot_label_distribution, plot_confusion_matrix
from sklearn import svm
from sklearn.metrics import classification_report
import pickle

# ------------------------------------------------------------
# model to predict on downdog (label 1)
# ------------------------------------------------------------

# ------------------------------------------------------------
# preprocess data
# ------------------------------------------------------------

# 0) load kaggle-dataset:
# ref: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
# kaggle_yoga_url = 'https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/download'
# ...


# # 1 ) extract landmarks: ---------------------------------------------------------------------------
# ut.extract_landmarks(f'./data/train/downdog/', 'data_preprocessed/train/', 'downdog', label=1)
# ut.extract_landmarks(f'./data/test/downdog/', 'data_preprocessed/test/', 'downdog', label=1)
#
# ut.extract_landmarks(f'./data/train/goddess/', 'data_preprocessed/train/', 'goddess', label=0)
# ut.extract_landmarks(f'./data/test/goddess/', 'data_preprocessed/test/', 'goddess', label=0)
#
# ut.extract_landmarks(f'./data/train/plank/', 'data_preprocessed/train/', 'plank', label=0)
# ut.extract_landmarks(f'./data/test/plank/', 'data_preprocessed/test/', 'plank', label=0)
#
# ut.extract_landmarks(f'./data/train/tree/', 'data_preprocessed/train/', 'tree', label=0)
# ut.extract_landmarks(f'./data/test/tree/', 'data_preprocessed/test/', 'tree', label=0)
#
# ut.extract_landmarks(f'./data/train/warrior2/', 'data_preprocessed/train/', 'warrior2', label=0)
# ut.extract_landmarks(f'./data/test/warrior2/', 'data_preprocessed/test/', 'warrior2', label=0)


# # 2 ) combine train data: ---------------------------------------------------------------------------
# downdog_train = pd.read_csv("data_preprocessed/train/dataset_downdog.csv")
# goddess_train = pd.read_csv("data_preprocessed/train/dataset_goddess.csv")
# plank_train = pd.read_csv("data_preprocessed/train/dataset_plank.csv")
# tree_train = pd.read_csv("data_preprocessed/train/dataset_tree.csv")
# warrior2_train = pd.read_csv("data_preprocessed/train/dataset_warrior2.csv")
#
# length = downdog_train.shape[0]
# distributed_length = int(length / 4)
#
# # combine all for generalisation/negative-examples
# combined = pd.concat([downdog_train, goddess_train[:distributed_length], plank_train[:distributed_length],
#                       tree_train[:distributed_length], warrior2_train[:distributed_length]], ignore_index=True)
# print(combined)
# # shuffle rows to randomize input later and reset indexation
# combined = combined.sample(frac=1).reset_index(drop=True)
# combined = combined.iloc[:, 1:]
#
# #save the combined training data
# combined.to_csv(f"data_preprocessed/train/combined.csv")


# # 3 ) combine test data: ---------------------------------------------------------------------------
# downdog_test = pd.read_csv("data_preprocessed/test/dataset_downdog.csv")
# goddess_test = pd.read_csv("data_preprocessed/test/dataset_goddess.csv")
# plank_test = pd.read_csv("data_preprocessed/test/dataset_plank.csv")
# tree_test = pd.read_csv("data_preprocessed/test/dataset_tree.csv")
# warrior2_test = pd.read_csv("data_preprocessed/test/dataset_warrior2.csv")
#
# length = downdog_test.shape[0]
# distributed_length = int(length / 4)
#
# # combine all for generalisation/negative-examples
# combined = pd.concat([downdog_test, goddess_test[:distributed_length], plank_test[:distributed_length],
#                       tree_test[:distributed_length], warrior2_test[:distributed_length]], ignore_index=True)
# # shuffle rows to randomize input later and reset indexation
# combined = combined.sample(frac=1).reset_index(drop=True)
# combined = combined.iloc[:, 1:]
# # save the combined training data
# combined.to_csv(f"data_preprocessed/test/combined.csv")


# # 4 ) Final preparations: ---------------------------------------------------------------------------
# data = pd.read_csv(f"data_preprocessed/train/combined.csv")
# datatest = pd.read_csv(f"data_preprocessed/test/combined.csv")
#
# x_train, y_train = data.iloc[:, 1:-1], data.iloc[:, -1]
# x_test, y_test = datatest.iloc[:, 1:-1], datatest.iloc[:, -1]
#
# print(x_train, y_train, x_test, y_test)
#
# plot_label_distribution(y_train, classes=['no Downdog', 'Downdog'])


# # 5 ) Create Model: ---------------------------------------------------------------------------
# model = svm.SVC(kernel='poly')
# model.fit(x_train,y_train)
# prediction = model.predict(x_test)
# target_names = ['0', '1']
# y_val_new = y_test.astype('uint8').tolist()
# prediction_new = prediction.astype('uint8').tolist()
# print(y_val_new)
# print(prediction_new)
# print(classification_report(y_val_new, prediction_new, target_names=target_names))
# plot_confusion_matrix(y_val_new, prediction_new, target_names, title='Confusion Matrix', cmap=None, normalize=True,
#                       decimals=2)

# 6 ) Serialize Model: ---------------------------------------------------------------------------
# pickle.dump(model, open('model.pkl','wb'))


# ===================== START HERE FOR TESTING PURPOSES: =====================
# Loading model
model = pickle.load(open('model.pkl', 'rb'))

# 7 ) Test on live video:
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

frame = 0
max_frames = 5000
res = ''

green = (0, 204, 0)
red = (0, 0, 255)

with mp_pose.Pose(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        temp = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, j in zip(ut.points, landmarks):
                temp = temp + [j.x, j.y, j.z]
            y = model.predict([temp])
            if y == 1:
                # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
                image = cv2.putText(image, "downdog", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 4)
            if y == 0:
                image = cv2.putText(image, "not downdog", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 4)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# ------------------------------------------------------------
if __name__ == '__main__':
    print('PyCharm')
# ------------------------------------------------------------
