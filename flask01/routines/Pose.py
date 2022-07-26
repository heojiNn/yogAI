import evaluation.evaluation_util as eval_ut
from os.path import exists
import cv2
import routines.util as ut
static_img_path = f'static/data/'


class Pose:
    def __init__(self, path, duration, breath_dur, static, fps=0, reprocess_data=False):
        self.path = path
        self.project_rel_path = static_img_path + path
        self.processed_path = "static/processed/" + path.split(sep='.')[0] + '.txt'
        self.duration = duration  # seconds (used in gen_pose_img() to determine yield)
        self.breath_dur = breath_dur  # breath hold duration
        self.static = static  # update: dynamic pose = transition, now
        self.reprocess_data = reprocess_data
        self.data = None
        if not self.static:
            self.FPS = fps
            self.load_data()
            #self.data = eval_ut.get_frames_from_vid(filepath=self.project_rel_path)
        else:
            self.FPS = None
            #self.data = None


    def gen_data(self):
        print("No preprocessed Data found!")
        print(f"Generating Data for: {self.path}")
        priv_cam = cv2.VideoCapture(self.project_rel_path)
        #self.FPS = priv_cam.get(cv2.CAP_PROP_POS_FRAMES)
        pts = []
        range_num = int(priv_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(range_num):
            suc, img = priv_cam.read()
            pts.append(ut.generate_pts(image=img))
            print(f"Progress generating Data: {i}/{range_num} Frames = {100*i/range_num} %")
        ut.save_pts(path=self.processed_path, data=pts)
        print(f"Finished generating Data for: {self.path}\nSaved as: {self.processed_path}")

    def load_data(self):
        if exists(self.processed_path) and not self.reprocess_data:
            print(f"Loading data for {self.path} from {self.processed_path}")
            self.data = ut.load_pts(self.processed_path)
            print("Loading complete!")
            return
        self.gen_data()
        self.load_data()

    def select_spec_frame(self, pos):
        #print(f"selected data at of {self.path} at {pos}")
        print(self.data[int(pos)])
        return self.data[int(pos)]
