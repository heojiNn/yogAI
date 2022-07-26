import os
import time

import cv2
import numpy as np
import time as tm
import routines.util as util

static_img_path = f'static/data/'
images = os.listdir(static_img_path)
images = np.array([x for x in images])
length = len(images)


class WorkoutManager:
    def __init__(self):
        self.workout = None
        self.image_index = 0
        self.__img_path__ = static_img_path + images[self.image_index]  # path to pose in active workout
        self.__tr_path__ = static_img_path  # path to transition in active workout
        self.reference = util.generate_pts(path=self.__img_path__)  # reference to landmarks of active pose
        self.kill = False  # flag to mark state of workout execution
        self.transition = False  # flag to mark state traversals (transition->pose->transition)
        # testing purposes
        self.video_path = self.__img_path__ + ""
        self.cam = None
        self.data = None  # saves all frames of dynamic pose in list
        self.finished = False

    def has_active(self):
        if self.workout is None:
            return False
        return True

    def change_workout(self, workout):
        self.workout = workout

    def get_cam(self):
        pose = self.workout.get_pose()
        if pose.static:
            print("Keine Dynamische Pose!")
            return None
        return self.cam

    def load_spec_image(self, pos):
        if self.data is None or pos >= self.data.__len__:
            return False
        return self.data[pos]

    def update_pose(self):
        if self.workout is None:
            return
        pose = self.workout.get_pose()
        self.__img_path__ = static_img_path + pose.path
        self.image_index = int(np.where(images == pose.path)[0])
        if pose.static:
            self.reference = util.generate_pts(path=self.__img_path__)
        else:
            # TODO implement for non Static poses
            self.video_path = self.__img_path__
            # self.gen = eval_ut.get_frames_from_vid(filepath=self.video_path)
            self.data = pose.data
            # self.workout.poses[self.workout.isNext-1].data =

    def update_transition(self):
        if self.workout is None or self.workout.get_transition() is None:
            return
        transition = self.workout.get_transition()
        self.__tr_path__ = static_img_path + transition.path

    def breathe(self, pose):
        timer = 0
        while timer < pose.duration and not self.finished:
            # print("Breath in!")
            tm.sleep(0.5)
            # print("Hold!")
            tm.sleep(pose.breath_dur)
            # print("breath out!")
            tm.sleep(0.5)
            timer += 1 + pose.breath_dur
        return

    def start(self):
        if self.workout is None:
            return Exception('No workout selected')

        self.finished = False
        self.kill = False

        while self.workout.has_next_pose() and not self.kill:
            # Update transition paths:
            if self.workout.get_transition() is not None:
                self.transition = True
                transition = self.workout.get_transition()
                self.workout.next_transition()
                print(f"Nächste Transition: {transition.path}!")
                tm.sleep(transition.get_duration())  # wait for duration before next update schedule
                self.transition = False
                self.update_transition()

            # Update pose paths:
            self.workout.next_pose()
            pose = self.workout.get_pose()
            print(f"Nächste Pose: {pose.path}!")
            self.update_pose()
            if pose.static:
                # runner for static poses
                self.breathe(pose)
                if self.kill:
                    print(f"Pose {pose.path} cancelled!")
                else:
                    print(f"Pose {pose.path} completed!")
            else:
                # runner for dynamic poses
                self.breathe(pose)
                print(f"Dynamic-Pose {pose.path} abgeschlossen!")

        self.finished = True
        # handle the case when workout has been cancelled (not finished), reset it:
        if self.kill:
            self.workout.reset()
            print("Workout cancelled")
        else:
            print("Workout finished!")

