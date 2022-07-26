from os.path import exists
import cv2
import routines.util as ut

tr_path = f'static/data/'


class Transition:
    def __init__(self, path, duration):
        self.path = tr_path + path
        self.duration = duration  # seconds

    def get_duration(self):
        return self.duration
