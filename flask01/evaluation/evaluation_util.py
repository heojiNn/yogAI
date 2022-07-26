import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import cv2
from moviepy.editor import VideoFileClip, clips_array, ImageSequenceClip, CompositeVideoClip, ColorClip
import numpy as np

'''
Entweder ein bereits extrahiertes video als vidcap übergeben, oder einen filepath zum Video. Returns List of frames
'''


def get_frames_from_vid(vidcap, filepath=None):
    frames = []
    if filepath:
        vidcap = cv2.VideoCapture(filepath)
    success, image = vidcap.read()
    print(image, success)
    count = 0
    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        frames.append(image)
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print(f"{count} frames were extracted.")
    return frames


"""
    Create an evaluation video with the target pose in the top left, the video in the top right and an animation of the
    data on the bottom.

    Parameters data, images, poses, fps, output_dir
    len(data[0][0]) == len(images) == len(poses)
    ----------
    data : array[tuple(array, string)]
        array of data entries you'd like to plot and their label. 
        So if you want to plot scores for LEFT_WRIST, RIGHT_WRIST you would set data to 
        [([left_wrist_data], 'LEFT_WRIST'), ([right_wrist_data], 'RIGHT_WRIST')]

    images : array of frames from the captured video

    poses : array of frames of the reference poses

    fps : the fps you would like your evaluation video to run
        the resulting length of the video will be len(images) / fps

    output_dir : the directory and name of your eval vid
        it makes sense to do something like output_dir = dir + datetime.now for unique names

    Returns
    -------
    no return value but an animation file and the evaluation video are created and saved
    """


def generate_eval_vid(data, images, poses, fps, output_dir):
    # TODO: assert params are correct
    frames = len(images)
    nb_plots = len(data.keys())

    x_data = [[] for _ in range(nb_plots)]
    y_data = [[] for _ in range(nb_plots)]
    lines = []
    fig, ax = plt.subplots(figsize=(32, 9), dpi=60)
    ax.set_xlim(0, frames)
    ax.set_ylim(0, 1)
    for key, entry in data.items():
        line, = ax.plot(0, 0)
        line.set_label(key)
        lines.append(line)
    ax.legend(loc='upper right')

    def animate(i):
        for key, x, y, line in zip(data, x_data, y_data, lines):
            x.append(i)
            y.append(data[key][i])
            line.set_xdata(x)
            line.set_ydata(y)
        return lines,

    itv = 1000 / fps
    input_frames = range(frames)
    ani = FuncAnimation(fig, animate, interval=itv, frames=input_frames)
    mywriter = animation.FFMpegWriter(fps=fps)
    ani_path = output_dir + '_ani.mp4'
    ani.save(ani_path, writer=mywriter)

    # height, width, layers = images[0].shape
    # images_resized = [Image.fromarray(img).resize((width, height)) for img in images]
    # width, height, layers = poses[0].shape
    # poses_resized = [Image.fromarray(img).resize((width, height)) for img in poses]
    width, height = 960, 540
    print(f"width: {width} height: {height}")
    poses_resized = [cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2RGB) for img in poses]
    images_resized = [cv2.resize(img, (width, height)) for img in images]

    # generate video from images:
    base_clip = ColorClip((1920, 1080), color=[0, 0, 0], duration=frames / fps)

    clip1 = ImageSequenceClip(poses_resized, fps=fps)
    clip2 = ImageSequenceClip(images_resized, fps=fps)
    clip3 = VideoFileClip(ani_path)

    # split screen
    # final_clip = clips_array([[clip3, clip2],
    #                           [clip1, ]])
    final_clip = CompositeVideoClip([base_clip,
                                     clip1.set_position(("left", "top")),
                                     clip2.set_position(("right", "top")),
                                     clip3.set_position(("center", "bottom"))])
    combined_path = output_dir + '_eval.mp4'
    final_clip.write_videofile(combined_path)


def test_animation(data, output_dir, fps):
    frames = len(data[0][0])
    nb_plots = len(data)

    x_data = [[] for _ in range(nb_plots)]
    y_data = [[] for _ in range(nb_plots)]
    lines = []
    fig, ax = plt.subplots(figsize=(32, 9), dpi=60)
    ax.set_xlim(0, frames)
    ax.set_ylim(0, 1)
    for entry in data:
        line, = ax.plot(0, 0)
        line.set_label(entry[1])
        lines.append(line)
    ax.legend(loc='upper right')

    def animate(i):
        for j, (x, y, line) in enumerate(zip(x_data, y_data, lines)):
            x.append(i)
            y.append(data[j][0][i])
            line.set_xdata(x)
            line.set_ydata(y)
        return lines,

    itv = 1000 / fps
    input_frames = range(frames)
    ani = FuncAnimation(fig, animate, interval=itv, frames=input_frames)
    mywriter = animation.FFMpegWriter(fps=fps)
    ani_path = output_dir + '_ani.mp4'
    ani.save(ani_path, writer=mywriter)


if __name__ == '__main__':
    frames = 100
    data1 = np.random.uniform(low=0.0, high=1., size=(frames,)), 'data1'
    data2 = np.random.uniform(low=0.0, high=1., size=(frames,)), 'data2'
    data3 = np.random.uniform(low=0.0, high=1., size=(frames,)), 'data3'
    data4 = np.random.uniform(low=0.0, high=1., size=(frames,)), 'data4'
    data = [data1, data2, data3, data4]
    # test_animation(data, 'gifs/test_multiple6', 10)

    video = get_frames_from_vid(None, filepath='gifs/test_multiple2_ani.mp4')
    poses = get_frames_from_vid(None, filepath='gifs/test_multiple3_ani.mp4')
    generate_eval_vid(data, video, poses, 10, 'gifs/test_3_split_screen_5')
