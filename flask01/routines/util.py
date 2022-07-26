import math
import time
import datetime
from timeit import timeit

import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_norm = mp.solutions.drawing_utils.DrawingSpec


verbose = False
in_person_test = False  # mirrors image horizontally if True

# ------------------------------------------------------------#

blur = 5
camFPS = 1
# best_threshold = 0.2
# medium_threshold = 0.35

i = 0
j = 0

used_landmarks = [
    # mp_pose.PoseLandmark.NOSE,
    # mp_pose.PoseLandmark.LEFT_EYE,
    # mp_pose.PoseLandmark.RIGHT_EYE,
    # mp_pose.PoseLandmark.LEFT_EAR,
    # mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    # mp_pose.PoseLandmark.LEFT_PINKY,
    # mp_pose.PoseLandmark.RIGHT_PINKY,
    # mp_pose.PoseLandmark.LEFT_INDEX,
    # mp_pose.PoseLandmark.RIGHT_INDEX,
    # mp_pose.PoseLandmark.LEFT_THUMB,
    # mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    # mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    # mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

test_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER
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


def normalize_pos(pts, gt=False, ref=0):  # moves pose into normalized position
    lr, rr = None, None
    if gt:
        if ref == 0:
            # get positions in pts of Left hip and right hip respectively
            lr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.LEFT_HIP)[0])
            rr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.RIGHT_HIP)[0])
            # get left and right reference points
            lr = pts[lr_ind]
            rr = pts[rr_ind]
        elif ref == 1:
            # get positions in pts of Left hip and right hip respectively
            lr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.LEFT_SHOULDER)[0])
            rr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
            # get left and right reference points
            lr = pts[lr_ind]
            rr = pts[rr_ind]
    else:
        # get positions in pts of Left hip and right hip respectively
        lr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.LEFT_HIP)[0])
        rr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.RIGHT_HIP)[0])
        # get left and right reference points
        lr = pts[lr_ind]
        rr = pts[rr_ind]
        ref = 0
        if lr is None or rr is None:  # use shoulder where hip is not found
            lr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.LEFT_SHOULDER)[0])
            rr_ind = int(np.where(np.array(used_landmarks) == mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])

            lr = pts[lr_ind]
            rr = pts[rr_ind]
            ref = 1

    if lr is None or rr is None:
        return [None for n in used_landmarks], -1
    norm_point = []
    for c1, c2 in zip(lr, rr):
        norm_point.append((c1 + c2) / 2)

    out = []
    for point in pts:
        pt = []
        for x1, x2 in zip(point, norm_point):
            pt.append(x1 - x2)
        out.append(pt)

    return out, ref  # ref=o: hips as reference, ref=1: shoulders as reference


def vabs(v):
    abs = 0
    for c in v:
        abs += c ** 2
    return abs ** 0.5


def dist_vec(v1, v2):
    vec = []
    for c1, c2 in zip(v1, v2):
        vec.append(abs(c2 - c1))
    return vec


def avg(list):  # expects list of numbers, returns avg of said numbers. Handles None values as trivially non existent
    sum = 0
    ind = 0
    for item in list:
        if item is None:
            continue
        sum += item
        ind += 1
    return sum / ind


def normalize_size(inp, ref):
    # Calculate Scaling factor

    # NormVectors with respect to poseinvariant vectors wrist-elbow, elbow-shoulder, knee-hip, ankle-knee
    Norm_pairs = [[mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW],
                  [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW],
                  [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER],
                  [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER],
                  [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP],
                  [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP],
                  [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE],
                  [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE]]

    scaling_candidates = []

    for norms in Norm_pairs:
        norm1 = norms[0]
        norm2 = norms[1]

        vec1_ind = int(np.where(np.array(used_landmarks) == norm1)[0])
        vec2_ind = int(np.where(np.array(used_landmarks) == norm2)[0])

        inp_vec1 = inp[vec1_ind]
        inp_vec2 = inp[vec2_ind]
        ref_vec1 = ref[vec1_ind]
        ref_vec2 = ref[vec2_ind]

        if inp_vec1 is None or inp_vec2 is None:
            scaling_candidates.append(None)
            break

        inp_distv = dist_vec(inp_vec1, inp_vec2)
        inp_abs = vabs(inp_distv)
        ref_distv = dist_vec(ref_vec1, ref_vec2)
        ref_abs = vabs(ref_distv)

        normfactor = ref_abs / inp_abs
        scaling_candidates.append(normfactor)

    normscalar = avg(scaling_candidates)
    if verbose: print(f"Skalierung: {normscalar}")

    # Scale landmarks accordingly to Scaling factor with respect to zero vector | Mathematically speaking this scaling should work just fine, I proved it
    outp = []

    for lm in inp:
        if lm is None:
            outp.append(None)
            continue
        for c in lm:
            c *= normscalar
        outp.append(lm)

    return outp


def compare_ptwise_static(inp, gt):  # inp: input, gt: ground-truth
    # compares inp vs gt normalized with respect to ref. Uses euclidean dist. Comparison on normalized input with
    # respect to size and position

    # normalize inp and gt
    # for position
    inp, ref = normalize_pos(inp)
    gt, _ = normalize_pos(gt, gt=True, ref=ref)
    # for size
    inp = normalize_size(inp, gt)

    comp = []
    # calc euclidean dist. of each landmark
    for point1, point2 in zip(inp, gt):
        if point1 is None or point2 is None:
            comp.append(None)
            continue
        score = 0
        for x1, x2 in zip(point1, point2):
            if x1 is None or x2 is None:
                comp.append(None)
                continue
            score += (x1 - x2) ** 2
        score = score ** 0.5
        comp.append(score)
    return comp  # returns distance of each landmark pair, distance is None if not calculable


def calc_vel(inp, timestamps):
    try:
        inp1 = inp[0]
        inp2 = inp[1]

        time1 = timestamps[0]
        time2 = timestamps[1]

        dt = (time2 - time1)
        vecs = []
        for lm1, lm2 in zip(inp1, inp2):
            if lm1 is None or lm2 is None:
                vecs.append(None)
                continue
            vel = []
            for c1, c2 in zip(lm1, lm2):
                vel.append(round((c2 - c1) / dt, blur))
            vecs.append(vel)
        return vecs
    except Exception:
        return [None for lm in inp1]


def do_speed(inp):
    try:
        outp = []
        for lm in inp:
            if lm is None:
                outp.append(None)
                continue
            speed = 0
            for c in lm:
                speed += c ** 2
            speed = math.sqrt(speed)
            outp.append(round(speed, blur))
        return outp
    except Exception:
        return [None for lm in inp]


def norm_vec(inp, speeds):
    try:
        outp = []
        for lm, sp in zip(inp, speeds):
            if lm is None or sp is None:
                outp.append(None)
                continue
            outv = []
            for c in lm:
                outv.append(round(c / sp, blur))
            outp.append(outv)
        return outp
    except Exception:
        return [None for lm in inp]


def compare_ptwise_dynamic(inp, gt, gt_FPS):
    # TODO calculate velocity-vector for inp and gt | finished?
    # remove unwanted blur by rounding up to <blur> decimals
    t_start = datetime.datetime.now()
    inputs = []
    for inp__ in inp:
        inp_ = []
        for entry in inp__:
            if entry is None:
                inputs.append(None)
                continue
            ent = []
            for c in entry:
                ent.append(round(c, blur))
            inp_.append(ent)
        inputs.append(inp_)
    ground_truth = []
    for gt__ in gt:
        gt_ = []
        for entry in gt__:
            if entry is None:
                ground_truth.append(None)
                continue
            ent = []
            for c in entry:
                ent.append(round(c, blur))
            gt_.append(ent)
        ground_truth.append(gt_)

    inp_vel = calc_vel(inputs, (1, 1 + 1 / camFPS))
    gt_vel = calc_vel(ground_truth, (1, 1 + 1 / gt_FPS))

    # TODO normalize velocity vectors and calculate speed | finished?

    inp_speeds = do_speed(inp_vel)
    gt_speeds = do_speed(gt_vel)

    inp_norm = norm_vec(inp_vel, inp_speeds)
    gt_norm = norm_vec(gt_vel, gt_speeds)

    # TODO compare speed & unit-vectors | finished?
    # Compare speeds
    rel_speed_diff = []
    abs_speed_diff = []

    for inp_speed, gt_speed in zip(inp_speeds, gt_speeds):
        #print(f"inp_speed = {inp_speed} | gt_speed = {gt_speed}")
        rel_speed_diff.append(inp_speed / gt_speed)
        abs_speed_diff.append(inp_speed - gt_speed)

    # compare unit-vectors by their distance on unit-sphere
    unit_dist = []
    for lm1, lm2 in zip(inp_norm, gt_norm):
        dist = 0
        if lm1 is None or lm2 is None:
            unit_dist.append(None)
            continue
        for c1, c2 in zip(lm1, lm2):
            dist += (c1 - c2) ** 2
        dist = math.sqrt(dist)
        unit_dist.append(dist)

    # compare unit-vectors by their angle
    angles = [0 for lm in used_landmarks]
    # for lm1, lm2 in zip(inp_norm, gt_norm):
    #     if lm1 is None or lm2 is None:
    #         angles.append(None)
    #         continue
    #     # calculate scalar-product
    #     scalar = 0
    #     for c1, c2 in zip(lm1, lm2):
    #         scalar += c1 * c2
    #     angle = ((math.acos(scalar)) % (2*math.pi)) * (180/math.pi)
    #     angles.append(angle)
    t_end = datetime.datetime.now()
    td = t_end - t_start

    # print("-" * 30 + t_start.strftime("%s") + 30 * "-")
    # print("-" * 30 + t_end.strftime("%s") + 30 * "-")
    # print("-" * 30 + str(td.microseconds) + 30 * "-")
    print("-" * 30 + f"\n\n\nTIMEDELTA{td}\n\n\n" + "-" * 30)
    return rel_speed_diff, abs_speed_diff, unit_dist, angles

def rotation_matrix_from_vectors(vec1, vec2): # returns matrix to redirect vec1 to vec2

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a,b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def calc_animation(inp, std): # inp: scanned pose of human | std: standard pose angles of virtual char  --- returns rotation matrices to update virtual chars pose
    # Angles to update between: elbow-wrist, shoulder-elbow, shoulder-hip, hip-knee, knee-ankle
    angle_pairs = [[mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
                   [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
                   [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
                   [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
                   # [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER],
                   # [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER],
                   # [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],
                   # [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
                   # [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
                   # [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
                   ]

    rotation_matrices = []

    for j, (lm1, lm2) in enumerate(angle_pairs):
        ind1 = int(np.where(np.array(used_landmarks) == lm1)[0])
        ind2 = int(np.where(np.array(used_landmarks) == lm2)[0])

        p1 = np.array(inp[ind1])
        p2 = np.array(inp[ind2])

        if p1 is None or p2 is None:
            rotation_matrices.append(np.eye(4))
            continue

        v_inp = p2 - p1
        v_cur = std[j]

        rot_mat = rotation_matrix_from_vectors(v_cur, v_inp)
        rotation_matrices.append(rot_mat)

    return rotation_matrices


# ------------------------------------------------------------#


def save_pts(path, data):
    # saves data into path, expects triencapsuled list or tuple like [DataEntry][Point][Coordinate]
    # saves NoneType point as "N/V;", skips NoneType entries
    with open(path, mode="w", encoding='utf-8') as file:
        for line in data:
            # print(str(line))
            d = ""
            if line is None:
                for none in used_landmarks:
                    d += "N/V;"
                file.write("%s\n" % d)
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


def generate_pts(image=None, path=None):
    # load img from path, process with blazepose and return list with coordinates of used_landmarks as (x, y, z) tuple
    # if no image or no landmarks are found under path None is returned

    # with mp_pose.Pose(static_image_mode=True, model_complexity=2,
    #                   enable_segmentation=True, min_detection_confidence=0.5) as pose:
    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.6) as pose:
        if image is None:
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


if __name__ == '__main__':
    t1 = np.array([0,1,0])
    t2 = np.array([-1, 4, 0])
    print(np.linalg.norm(t1))
    rot = np.dot(rotation_matrix_from_vectors(t1, t2), t1)
    print(np.linalg.norm(rot))
    print(rot/np.linalg.norm(rot))
    print(t2/np.linalg.norm(t2))
    print(50*"-")
    print("RotationsMatrix:", end="\n\n")
    rotmat = rotation_matrix_from_vectors(t1, t2)
    for line in rotmat:
        for el in line:
            print(f"{el}", end=",")
        print("0,")
    print("0,0,0,1")

    print("\nTransponiert: (wird benutzt für 3D Model)", end="\n\n")
    rotmat = np.transpose(rotmat)
    for line in rotmat:
        for el in line:
            print(f"{el}", end=",")
        print("0,")
    print("0,0,0,1")
    #print(rotmat_print)
    # Test = [[1, 2, 3] for i in range(30)]
    # Test2 = [[1, 2, 3] for j in range(30)]
    # Speeds = [0.2 for k in range(30)]
    #
    # Test_ = [[[1, 2, 3] for i in range(30)], [[2, 3, 4] for i in range(30)]]
    # Ref = [[[1, 2, 3] for i in range(30)], [[2, 3, 4] for i in range(30)]]
    #
    # print(timeit("calc_vel((Test, Test2), (1, 1.01))", "from __main__ import calc_vel, Test, Test2", number=10000))
    # print(timeit("do_speed(Test)", "from __main__ import do_speed, Test, Test2", number=10000))
    # print(timeit("norm_vec(Test, Speeds)", "from __main__ import norm_vec, Test, Speeds", number=10000))
    # print(timeit("compare_ptwise_dynamic(Test_, Ref, 30)", "from __main__ import compare_ptwise_dynamic, Test_, Ref",
    #              number=10000))
    # print(calc_vel((Test,Test2), (1, 1.01)))


# ------------------------------------------------------------#
# Philipp:


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_used_landmarks(landmarks):
    used_lm = []
    for lm in used_landmarks:
        used_lm.append((landmarks.pose_landmarks.landmark[lm].x,
                        landmarks.pose_landmarks.landmark[lm].y,
                        landmarks.pose_landmarks.landmark[lm].z))
    return used_lm


def highlight_landmarks(image, landmarks, distances, best_threshold, medium_threshold, dim=[640, 480]):
    """ Draws a cirlce for each used landmarks on the skeleton. Coloration based on the distance thresholds."""
    for lm, item in zip(used_landmarks, distances):
        if item <= best_threshold:
            cv2.circle(image,
                       tuple(
                           np.multiply([landmarks[lm].x, landmarks[lm].y], dim).astype(int)),
                       2, (0, 255, 0), 10)
        elif item <= medium_threshold:
            cv2.circle(image,
                       tuple(
                           np.multiply([landmarks[lm].x, landmarks[lm].y], dim).astype(int)),
                       2, (0, 165, 255), 10)
        else:
            cv2.circle(image,
                       tuple(
                           np.multiply([landmarks[lm].x, landmarks[lm].y], dim).astype(int)),
                       2, (0, 0, 255), 10)


def highlight_denormalized_landmarks(image, landmarks, distances, dim, best_threshold, medium_threshold):
    """ Draws a cirlce for each used landmarks on the skeleton. Coloration based on the distance thresholds."""
    for lm, item in zip(used_landmarks, distances):
        x_px = min(math.floor(landmarks[lm].x * dim[1]), dim[1] - 1)
        y_px = min(math.floor(landmarks[lm].y * dim[0]), dim[0] - 1)
        if item <= best_threshold:
            cv2.circle(image,
                       tuple([x_px, y_px]),
                       2, (0, 255, 0), 10)
        elif item <= medium_threshold:
            cv2.circle(image,
                       tuple([x_px, y_px]),
                       2, (0, 165, 255), 10)
        else:
            cv2.circle(image,
                       tuple([x_px, y_px]),
                       2, (0, 0, 255), 10)


def highlight_blazepose_landmarks(image, results, best_threshold, medium_threshold, current_dist=None):
    # TODO: find a way to extract landmarks out of the 'NormalizedLandmarkList' from mediapipe
    # mediapipe.framework.formats.landmark.proto, to grant ability to draw individual landmarks directly

    for lm, item in zip(used_landmarks, current_dist):
        if item <= best_threshold:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1)
            )
        elif item <= medium_threshold:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 165, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(
                    color=(0, 165, 255), thickness=1, circle_radius=1)
            )
        else:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=1, circle_radius=1)
            )


def plot_pose_3d(results):
    mp_drawing.plot_landmarks(results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 0), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 0), thickness=1, circle_radius=1)),


def draw_pose_landmarks(image, results, color=(255, 255, 255), thickness=1, circle_radius=1):
    mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=color, thickness=thickness, circle_radius=circle_radius),
                    mp_drawing.DrawingSpec(
                        color=color, thickness=thickness, circle_radius=circle_radius)
                )


def draw_holistic_landmarks(image, results, color=(255, 255, 255), thickness=1, circle_radius=1):
    mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=color, thickness=thickness, circle_radius=circle_radius),
                    mp_drawing.DrawingSpec(
                        color=color, thickness=thickness, circle_radius=circle_radius)
                )


def put_healthbar(image, hp_width, hp_height, health, thickness):
    """
        Put 2 rectangles on the image for each player.
        One in the back in red, one in the front in green.
        Green is 100 % width and starts decreasing exposing red.

        Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
    """
    red = (255, 0, 0)  # RGB
    green = (0, 255, 0)

    x = int(image.shape[1])
    y = int(image.shape[0])

    # first (red) rectangle (always same sized)
    top_left = int(x - hp_width * x), 0
    bottom_right = int(x * (1 - hp_width) + (health / 100) * x * hp_width), int(hp_height * y)
    bottom_origin = x, int(hp_height * y)

    image = cv2.rectangle(image, top_left, bottom_origin, red, thickness)
    image = cv2.rectangle(image, top_left, bottom_right, green, thickness)
    return image


def put_healthbar_mult(image, hp_width, hp_height, health1, health2, thickness):
    """
        Put 2 rectangles on the image for each player.
        One in the back in red, one in the front in green.
        Green is 100 % width and starts decreasing exposing red.

        Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
    """
    green = (0, 255, 0)  # RGB
    red = (0, 0, 255)

    # first (red) rectangle (always same sized)
    x = int(image.shape[1])
    y = int(image.shape[0])

    # left healthbar
    top_left = 0, 0
    top_left_hp = int(x * hp_width - (health1 / 100) * x * hp_width), 0
    bottom_origin = int(hp_width * x) - 2, int(hp_height * y)

    image = cv2.rectangle(image, top_left, bottom_origin, red, thickness)
    image = cv2.rectangle(image, top_left_hp, bottom_origin, green, thickness)

    # right healthbar
    top_left = int(x - hp_width * x), 0
    bottom_right = int(x * (1 - hp_width) + (health2 / 100) * x * hp_width), int(hp_height * y)
    bottom_origin = x, int(hp_height * y)

    image = cv2.rectangle(image, top_left, bottom_origin, red, thickness)
    image = cv2.rectangle(image, top_left, bottom_right, green, thickness)
    return image



