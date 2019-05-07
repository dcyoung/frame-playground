import os
import os.path as osp
import numpy as np
import cv2
import utils.easing as easing
import math

_DEFAULT_BACKGROUND_COLOR = (0, 0, 0)
_DEFAULT_TARGET_BACKGROUND_COLOR = (0, 255, 0)
_DEFAULT_FONT_COLOR = (255, 255, 255)
_DEFAULT_FRAME_SIZE = (300, 300)


def frame_with_centered_text(text, frame_size=None, background_color=None, font_color=None):
    """ Creates a dummy frame with text printed in the center.
    Arguments:
        text: text to draw in the middle of the frame
        frame_size: the output size of the generated frame (height, width)
        background_color: the background color of the frame
        font_color: the font color used to draw the text on the frame
    Returns:
        rgb image (numpy array) of size [height, width, 3]
    """
    if frame_size is None:
        frame_size = _DEFAULT_FRAME_SIZE
    if background_color is None:
        background_color = _DEFAULT_BACKGROUND_COLOR
    if font_color is None:
        font_color = _DEFAULT_FONT_COLOR

    frame_width, frame_height = frame_size
    # Create black frame
    img = np.zeros((frame_height, frame_width, 3), np.uint8)
    bg_r, bg_g, bg_b = background_color
    img[:, :, 0] = bg_r
    img[:, :, 1] = bg_g
    img[:, :, 2] = bg_b

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # get coords of bottom left corner based on desired center and text size
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    bottom_left_corner_of_text = (text_x, text_y)

    line_type = 2
    cv2.putText(img, text, bottom_left_corner_of_text,
                font, font_scale, font_color, line_type)

    return img


def numbered_frame(frame_number, frame_size=None, background_color=None, font_color=None):
    """ Creates a dummy frame with a number printed in the center.
    Arguments:
        frame_number: number to draw in the middle of the frame
        frame_size: the output size of the generated frame (height, width)
        background_color: the background color of the frame
        font_color: the font color used to draw the number on the frame
    Returns:
        rgb image (numpy array) of size [height, width, 3]
    """
    if frame_size is None:
        frame_size = _DEFAULT_FRAME_SIZE
    if background_color is None:
        background_color = _DEFAULT_BACKGROUND_COLOR
    if font_color is None:
        font_color = _DEFAULT_FONT_COLOR
    return frame_with_centered_text(str(frame_number), frame_size, background_color, font_color)


def numbered_frames(frame_targets, duration_sec, output_fps=24, blend_frames=1, frame_size=None, normal_background_color=None, target_background_color=None, font_color=None):
    """ Creates a dummy sequence of frames, displaying incrementing
        frame number and highlighting provided target frames.
    Arguments:
        frame_targets: the target indices of frames to highlight
        duration_sec: the desired duration of the output video
        output_fps: the desired fps of the output video
        blend_frames: the number of frames around a target to blend the highlight
        frame_size: the output size of the generated frame (height, width)
        normal_background_color: the background color of untargeted frames
        target_background_color: the background color of target frames
        font_color: the font color used to draw the number on the frame
    Returns:
        sequence of frames (rgb images), numpy array of size [n, height, width, 3]
    """
    if frame_size is None:
        frame_size = _DEFAULT_FRAME_SIZE
    if normal_background_color is None:
        normal_background_color = _DEFAULT_BACKGROUND_COLOR
    if target_background_color is None:
        target_background_color = _DEFAULT_TARGET_BACKGROUND_COLOR
    if font_color is None:
        font_color = _DEFAULT_FONT_COLOR

    number_of_frames = int(output_fps * duration_sec)
    frames = np.zeros((number_of_frames,) + frame_size + (3,), np.uint8)

    for i in range(number_of_frames):
        is_target_frame = False
        for ft in frame_targets:
            if abs(i - ft) <= blend_frames:
                is_target_frame = True
                break
        background_color = target_background_color if is_target_frame else normal_background_color
        frames[i] = numbered_frame(
            i, frame_size, background_color=background_color, font_color=font_color)
    return frames


def save_video(frames, output_file, output_fps=24):
    """ Saves a sequence of frames as a video file on disk.
    Arguments:
        frames: sequence of frames (rgb images), numpy array of size [n, height, width, 3]
        output_file: the path to the output video file
        output_fps: the desired framerate of the output video
    """
    if not osp.exists(osp.dirname(output_file)):
        os.makedirs(osp.dirname(output_file))

    num_frames, height, width, channels = frames.shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_file, fourcc, output_fps, frame_size)

    for i in range(num_frames):
        video.write(frames[i])
    cv2.destroyAllWindows()
    video.release()


def get_frame_targets_from_time_targets(time_targets_ms, output_fps=24):
    """ Gets frame targets from a list of time targets
    Arguments:
        time_targets_ms: the timestamps of desired frames
        fps: the desired output frame rate
    Returns:
        frame indices associated with each timestamp
    """
    return [(t * output_fps) // 1000 for t in time_targets_ms]


def get_time_targets_from_frame_targets(frame_targets, fps=24):
    """ Gets time targets from a list of frame targets
    Arguments:
        frame_targets: the index of frames
        fps: the input frame rate associated with the frame targets
    Returns:
        timestamps (ms) for each frame target
    """
    return [1000 * f/fps for f in frame_targets]


def generate_video_with_target_frames(output_file, frame_targets, duration_sec, output_fps=24, blend_frames=1):
    """ Generates a dummy video with highlighted target frames and saves it to disk.
    Arguments:
        output_file: the path to the output video file
        frame_targets: the target indices of frames to highlight
        duration_sec: the desired duration of the output video
        output_fps: the desired fps of the output video
        blend_frames: the number of frames around a target to blend the highlight
    """
    frames = numbered_frames(
        frame_targets, duration_sec, output_fps, blend_frames)

    save_video(frames, output_file, output_fps)


def generate_video_with_target_times_ms(output_file, time_targets_ms, duration_sec=5, output_fps=24, blend_frames=1):
    """ Generates a dummy video with highlighted target frames at the time targets and saves it to disk.
    Arguments:
        output_file: the path to the output video file
        time_targets_ms: the target timestamps (in ms) for which to highlight frames
        duration_sec: the desired duration of the output video
        output_fps: the desired fps of the output video
        blend_frames: the number of frames around a target to blend the highlight
    """
    frame_targets = get_frame_targets_from_time_targets(
        time_targets_ms, output_fps)
    generate_video_with_target_frames(
        output_file, frame_targets, duration_sec, output_fps, blend_frames)


def generate_video_with_target_times_seconds(output_file, time_targets_sec, duration_sec=5, output_fps=24, blend_frames=1):
    """ Generates a dummy video with highlighted target frames at the time targets and saves it to disk.
    Arguments:
        output_file: the path to the output video file
        time_targets_sec: the target timestamps (in seconds) for which to highlight frames
        duration_sec: the desired duration of the output video
        output_fps: the desired fps of the output video
        blend_frames: the number of frames around a target to blend the highlight
    """
    generate_video_with_target_times_ms(
        output_file, [t*1000 for t in time_targets_sec], duration_sec, output_fps, blend_frames)


def remove_off_pace_frames(frame_times_ms, output_fps=24):
    """ Identifies frame indices whose timestamps are in sync with the desired framerate
    Arguments:
        frame_times_ms: frame timestamps (in ms) for a sequence of frames
        output_fps: the desired framerate
    Returns:
        List of indices for the input timestamps that are in sync with the output framerate
    """
    # Frame indices to keep
    to_keep = []
    n = len(frame_times_ms)

    last_claimed_frame = 0
    prev_frame_delta = output_fps * 1000

    for f_idx in range(n):
        frame_target = frame_times_ms[f_idx] * output_fps / 1000.0
        nearest_whole_frame = int(round(frame_target))
        dist_from_whole_frame = math.fabs(frame_target - nearest_whole_frame)

        if last_claimed_frame != nearest_whole_frame:
            to_keep.append(f_idx)
            last_claimed_frame = nearest_whole_frame
        elif dist_from_whole_frame < prev_frame_delta:
            # replace the last frame with the current frame
            if not not to_keep:  # only remove if list is not empty
                to_keep.pop()
            to_keep.append(f_idx)

        prev_frame_delta = dist_from_whole_frame
    return to_keep


def test_frame_generation():
    """ Tests frame generation """
    fps = 72
    blend_frames = 1
    duration_sec = 5
    target_times_ms = [1000, 2000, 3000, 4000, 5000]
    output_file = "C:\\Users\\dcyoung\\Documents\\GitHub\\frame-playground\\output\\frame_generation_test.avi"
    generate_video_with_target_times_ms(
        output_file, target_times_ms, duration_sec, fps, blend_frames)


def test_frame_easing(output_file="C:\\Users\\dcyoun\\Documents\\GitHub\\frame-playground\\output\\easing_test.avi"):
    """ Tests frame easing """
    output_fps = 24
    blend_frames = 1
    video_duration_ms = 12000

    start_time_ms = 1000
    retarget_duration_before = 10000
    retarget_duration_after = 5000
    diff = retarget_duration_before - retarget_duration_after
    target_ms = [start_time_ms, retarget_duration_before]

    target_frames = get_frame_targets_from_time_targets(target_ms, output_fps)

    frames = numbered_frames(
        target_frames, video_duration_ms / 1000, output_fps, blend_frames)

    n, h, w, c = frames.shape
    retargeted_frame_times = []
    for f_index in range(n):
        # original frame time
        t_ms = 1000 * f_index/output_fps
        if t_ms <= start_time_ms:
            # Before timewarp
            retargeted_ms = t_ms
        elif t_ms <= (start_time_ms + retarget_duration_before / 2):
            # First half of timewarp
            # retargeted frame time
            retargeted_ms = easing.ease_out_exp(
                t_ms, start_time_ms, retarget_duration_after, retarget_duration_before, start_time_ms)
        elif t_ms <= (start_time_ms + retarget_duration_before):
            # Second half of timewarp
            # retargeted frame time
            retargeted_ms = easing.ease_in_exp(
                t_ms, start_time_ms, retarget_duration_after, retarget_duration_before, start_time_ms)
        else:
            # After timewarp
            retargeted_ms = t_ms - diff
        retargeted_frame_times.append(retargeted_ms)
    to_keep = remove_off_pace_frames(retargeted_frame_times, output_fps)

    for j in range(len(to_keep)):
        i = to_keep[j]
        print(i, int(1000 * i/output_fps), int(retargeted_frame_times[i]), int(
            retargeted_frame_times[i] - retargeted_frame_times[i-1]))

    os.makedirs(osp.dirname(output_file))
    save_video(frames[to_keep], output_file, output_fps)


if __name__ == '__main__':
    """ Docstring """
    test_frame_generation()
    test_frame_easing()
