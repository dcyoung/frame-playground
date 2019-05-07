import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from time import strftime
from moviepy.editor import VideoFileClip, ImageSequenceClip
from flow_generator import *
from flowIO import write_flo_file, read_flo_file
from tqdm import trange
from visualize import plot_2D_flow_field

################################################################################
# SMOOTH VALUES OVER FRAMES
################################################################################
# frame_height = 3
# frame_width = 3
# num_frames = 5
# a = np.zeros([num_frames, frame_height, frame_width], dtype=np.float32)

# a[num_frames // 2, frame_height//2, frame_width//2] = 1
# # print(a)

# b = ndimage.gaussian_filter1d(a, 0.35, 0)
# print(b)

################################################################################
# Generate flow video
################################################################################
video_file = "D:\\Storage\\Videos\\Destiny 2\\2_pt1.mp4"
# video_file = osp.join('..', 'test_resources', 'video', 'acro_1.mp4')

# output_dir = osp.join(
#     'output', 'scratch_{0}'.format(strftime("%Y%m%d-%H%M%S")))
output_dir = osp.join('output', 'scratch_20181221-164936')

# start_frame, num_frames = 50, 100
start_frame, num_frames = 0, None
flo_file = osp.join(output_dir, 'generated_flow_for_video.npy')
trained_model_path = osp.join('models', 'flownet2-s.npz')
height, width = 320, 576

print("Creating flow generator, with input dimensions: h={0}, w={1}...".format(
    height, width))
gen = FixedSizeFlowGenerator(trained_model_path, height, width)

print("Importing clip...")
video_clip = VideoFileClip(
    video_file, target_resolution=(None, width))

print("Pre-processing frames...")
# Read and pre-process frames
if num_frames is None:
    num_frames = int(video_clip.fps * video_clip.duration)
pre_processed_frames = np.zeros([num_frames, height, width, 3], np.uint8)
for i, f in enumerate(video_clip.iter_frames()):
    if i < start_frame: 
        continue
    elif i >= start_frame + num_frames:
        break
    pre_processed_frames[i-start_frame] = gen.pre_process_frame(f)

print("Finished reading video with size: {0}".format(
    pre_processed_frames.shape))

try:
    print("Attempting to read flow fields...")
    flow_fields = read_flo_file(flo_file)
except:
    print("Failed to read. Generating flow fields now...")
    # Generate flow fields
    num_flow_fields = num_frames-1
    flow_fields = np.zeros(
        [num_flow_fields, height, width, 2], dtype=np.float32)
    for i in trange(num_flow_fields):
        # Get frames
        left, right = pre_processed_frames[i], pre_processed_frames[i+1]

        # Generate flow field
        flow_fields[i] = gen.apply(left, right)

    print("Saving flow fields...")
    # Save flo
    write_flo_file(flow_fields, flo_file)










print("Calculating overlap errors...")

def calc_error(flow_field_1, flow_field_2):
    """ Docstring """
    return np.mean(np.linalg.norm(flow_field_2 - flow_field_1, axis=1))


def compare_field_sequences(seq_1, seq_2):
    """ Docstring """
    n1, h1, w1, c1 = seq_1.shape
    n2, h2, w2, c2 = seq_2.shape
    assert h1 == h2, "Height of input sequences do not match."
    assert w1 == w2, "Width of input sequences do not match."
    assert c1 == c2, "Depth of input sequences do not match."

    # guarantee that seq 1 is longer or equal in length to seq 2
    if n1 < n2:
        return compare_field_sequences(
            seq_1=seq_2,
            seq_2=seq_1)

    errors_by_overlap = []
    # slide the shorter sequence over the longer
    for move in range(n1 + 2):
        # window start idx
        k = move - n2 + 1

        if k < 0 or k > n1-n2:
            seq_1_temp = np.zeros((n2, h1, w1, c1), dtype=np.float32)
            for x in range(max(k, 0), min(n1, move+1)):
                seq_1_temp[x-k] = seq_1[x]
            errors_by_overlap.append(calc_error(seq_1_temp, seq_2))
        else:
            errors_by_overlap.append(calc_error(seq_1[k:k+n2], seq_2))

    return errors_by_overlap


s1 = flow_fields[940:990]
s2 = flow_fields[960:970]

errors_by_overlap = compare_field_sequences(s1, s2)
plt.plot(range(len(errors_by_overlap)), errors_by_overlap)
plt.show()
exit(1)














# Use the 99th percentile of the magnitude as a saturation value,
#  to avoid outliers skewing the color scheme
u = flow_fields[:, :, :, 0]
v = flow_fields[:, :, :, 1]
mag = np.sqrt(u*u + v*v)
saturation = np.percentile(mag.flatten(), 99)
# plt.hist(mag.flatten(), bins='auto')
# plt.show()

print("Visualizing flow fields...")
output_flow_frame_files = []
output_vis_dir = osp.join(output_dir, "flow_fields")
os.makedirs(output_vis_dir, exist_ok=True)
for i in trange(num_flow_fields):
    # Visualize frames and flow field
    fig = plot_2D_flow_field(flow_fields[i], pre_processed_frames[i], magnitude_saturation=saturation)
    output_flow_frame_file = osp.join(
        output_vis_dir, 'flow_field_{0}.jpg'.format(start_frame + i))
    output_flow_frame_files.append(output_flow_frame_file)
    plt.savefig(output_flow_frame_file, bbox_inches='tight')
    plt.close(fig)

output_clip = ImageSequenceClip(output_flow_frame_files, fps=video_clip.fps)
print(output_clip.fps)
output_clip.write_videofile(
    osp.join(output_dir, "output.mp4"), fps=video_clip.fps)