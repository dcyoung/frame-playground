import os
import os.path as osp

def generate_flow(image_a, image_b):
    """ Generates a flow field from image a to image b
    Arguments:
        image_a: rgb image represented by numpy array of size [height, width, 3]
        image_b: rgb image represented by numpy array of size [height, width, 3]
    Returns:
        Flow field as a 2 channel image representing float values for u and v... [height, width, 2]
    """
    raise NotImplementedError()

def generate_flow_for_video(file_path):
    """ Generates flow fields for each pair of subsequent frames in a video file...
    Arguments:
        file_path: path to video file
    Returns:
        stack of flow fields, 1 for each pair of subsequent frames in the video file
            Numpy array of size [n, height, width, 2] where n is the number of flow fields (should be 1 less than number of video frames)
            Each flow field is a 2 channel image representing float values for u and v... [height, width, 2]
    """
    raise NotImplementedError()


 