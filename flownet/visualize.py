import numpy as np
import matplotlib.pyplot as plt


def plot_2D_flow_field(flow_field, background_img=None, magnitude_saturation=None):
    """Docstring 
    Args:
        flow_filed: [height, width, 2] numpy float array, representing u and v values at each pixel
        background_image: a background image [height, width, 3] numpy array (optional) to use as a backdrop for plot
    """
    h, w, c = flow_field.shape
    x = np.arange(0, w)
    y = np.arange(0, h)

    u = flow_field[:, :, 0]
    v = flow_field[:, :, 1]

    # u and v are in img coords... but we need to plot in plt with normal coords
    # so mirror flip in the vertical direction, and negate the vertical (v) values
    u = np.flip(u, 0)
    v = -np.flip(v, 0)

    magnitude = np.sqrt(u*u + v*v)
    max_mag = magnitude.max()
    if magnitude_saturation is not None and max_mag > magnitude_saturation:
        magnitude[magnitude > magnitude_saturation] = magnitude_saturation
        max_mag = magnitude_saturation
    normalized_magnitude = magnitude / max_mag

    fig, ax = plt.subplots()
    if background_img is not None:
        ax.imshow(background_img, extent=[0, w, 0, h])
    pix_per_30_cell_density = 320.0
    strm = ax.streamplot(x, y, u, v, color=normalized_magnitude, linewidth=1 + normalized_magnitude,
                         cmap=plt.cm.rainbow, density=[w / pix_per_30_cell_density, h / pix_per_30_cell_density], arrowstyle='->', arrowsize=1.5)
    fig.colorbar(strm.lines)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    plt.axis('scaled')
    return fig


def fig_to_image(fig):
    """ Converts a Matplotlib figure to a 3D numpy array with RGB channels and return it
    Arguments:
        fig: a matplotlib figure
    Returns:
        numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf


if __name__ == '__main__':
    """ Docstring """
    import os.path as osp
    from flowIO import read_flo_file

    # for category in ['Dimetrodon', 'Grove2', 'Grove3', 'Hydrangea', 'RubberWhale', 'Urban2', 'Urban3', 'Venus']:
    for category in ['Hydrangea', 'Urban2', 'Urban3']:
        flo_file_path = osp.join(
            'eval', 'other-gt-flow', category, 'flow10.flo')
        flow_field = read_flo_file(flo_file_path)
        fig = plot_2D_flow_field(flow_field)
        plt.show()
        plt.close(fig)
