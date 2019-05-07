import os
import os.path as osp
import numpy as np

TAG_FLOAT = 202021.25
# TAG_STRING = b"PIEH"


class UnsupportedFileFormat(Exception):
    pass


def read_flo_file(file_path):
    """ Reads a flow field from disk. 
    Returns:
        flow field in the form of a 2 channel image (numpy array) 
        representing float values for u and v... [height, width, 2]
    """
    # Determine the extension
    _, extension = os.path.splitext(file_path)

    # Parse flow field from numpy saved array file format
    if extension.lower() == '.npy':
        return np.load(file_path)

    # Parse flow field from "flo" file format
    if extension.lower() == '.flo':
        with open(file_path, 'rb') as f:
            tag = np.fromfile(f, dtype=np.float32, count=1)[0]
            if tag != TAG_FLOAT:
                raise TypeError("File is not the apporpriate .flo format.")
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            data = np.fromfile(f, dtype=np.float32)

        return np.reshape(data, [height, width, 2])

    raise UnsupportedFileFormat(
        "File format {0} is not supported.".format(extension))


def write_flo_file(flow_field, file_path):
    """ Writes a flow field to disk. 
    Arguments:
        flow_field: flow field in the form of a 2 channel image (numpy array)
            representing float values for u and v... [height, width, 2]
        file_path: path to the output file
    """

    # Determine the extension
    _, extension = os.path.splitext(file_path)

    if extension.lower() not in ['.flo', '.npy']:
        raise UnsupportedFileFormat(
            "File format {0} is not supported.".format(extension))

    # Write flow field to numpy saved array file format
    if extension.lower() == '.npy':
        # make output directory if it doesn't yet exist
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        np.save(file_path, flow_field)

    # Write flow field to "flo" file format
    if extension.lower() == '.flo':
        # reshape flow field
        h, w, c = flow_field.shape
        flattened_flow_field = np.reshape(flow_field, [h * w * c])
        with open(file_path, 'wb') as f:
            np.array([TAG_FLOAT], dtype=np.float32).tofile(f)
            np.array([w, h], dtype=np.int32).tofile(f)
            flattened_flow_field.tofile(f)


if __name__ == '__main__':
    """ Docstring """
    import os
    import os.path as osp
    input_flo_file = osp.join('eval', 'other-gt-flow',
                              'Hydrangea', 'flow10.flo')
    flo_img = read_flo_file(input_flo_file)
    print(flo_img.shape)
