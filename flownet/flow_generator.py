import numpy as np
import cv2
from tensorpack import OfflinePredictor, PredictConfig, get_model_loader, imgaug
import flownet_models


def get_model_by_type(model_type):
    """ Gets a flownet model by type
    Arguments:
        model_type: type of flownet model. one of 'flownet2-s', 'flownet2-c' or 'flownet2'
    Returns:
        flownet model
    """
    available_models = {
        "flownet2-s": flownet_models.FlowNet2S,
        "flownet2-c": flownet_models.FlowNet2C,
        "flownet2": flownet_models.FlowNet2,
    }
    model = available_models.get(model_type, None)
    if model is None:
        raise ValueError("Unsupported model type: {0}".format(model_type))
    return model


class FixedSizeFlowGenerator:
    """ Docstring """

    def __init__(self, trained_model_path, height=384, width=768, model_type="flownet2-s"):
        """ Docstring
        Arguments:
            trained_model_path:
            height:
            width:
            model_type: type of flownet model. one of 'flownet2-s', 'flownet2-c' or 'flownet2'
        """
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("Height and Width must be mulitples of 64.")
        self.input_height = height
        self.input_width = width
        flownet_model = get_model_by_type(model_type)
        self.predict_func = OfflinePredictor(
            PredictConfig(
                model=flownet_model(height=self.input_height,
                                    width=self.input_width),
                session_init=get_model_loader(trained_model_path),
                input_names=["left", "right"],
                output_names=["prediction"],
            )
        )

    def pre_process_frame(self, image):
        """ Pre-prcoesses a frame for the model.
        Arguments:
            image: input image to pre-process
        Returns:
            pre-processed image
        """
        # # Smooth image
        # kernel_size = (5, 5)
        # image = cv2.blur(image, kernel_size)

        # # Quantize colors
        # num_clusters = 4
        # image = reduce_color_image(image, num_clusters)

        # resize such that 1 image dim completely matches the crop dim
        h, w, c = image.shape
        scale_factor = max(float(self.input_height)/h,
                           float(self.input_width)/w)
        resize_size = (int(h*scale_factor), int(w*scale_factor))
        aug = imgaug.Resize(resize_size, cv2.INTER_CUBIC)
        image = aug.augment(image)

        # crop
        aug = imgaug.CenterCrop((self.input_height, self.input_width))
        image = aug.augment(image)

        return image

    def apply(self, left, right):
        """ Docstring
        Arguments:
            left: rgb image represented by numpy array of size [height, width, 3]
            right: rgb image represented by numpy array of size [height, width, 3]
        Returns:
            Flow field as a 2 channel image representing float values for u and v... [height, width, 2]
        """
        h, w = left.shape[:2]
        if h != self.input_height or w != self.input_width:
            raise ValueError("")

        # Rearrange for channels first (HWC -> CHW)
        left_input, right_input = [
            x.astype("float32").transpose(2, 0, 1)[None, ...] for x in [left, right]
        ]
        return self.predict_func(left_input, right_input)[0].transpose(0, 2, 3, 1)[0]

def reduce_color_image(image, num_clusters=8):
    """ Quantizes the colors in the input image using k-means clustering.
    Arguments:
        image: input image to quantize
        num_clusters: number of colors/clusters to use for k-means based quantization
    Returns:
        an image quantized to num_clusters colors
    """
    from sklearn.cluster import MiniBatchKMeans
    import cv2

    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=num_clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant


if __name__ == "__main__":
    """ Docstring """
    import os
    import os.path as osp
    import matplotlib.pyplot as plt
    from visualize import plot_2D_flow_field

    # Read images
    frame_paths = [
        osp.join("images", "0.png"),
        osp.join("images", "1.png"),
        osp.join("images", "2.png"),
        osp.join("images", "3.png"),
        osp.join("images", "4.png"),
        osp.join("images", "5.png"),
        osp.join("images", "6.png"),
    ]

    # Create generator
    trained_model_path = osp.join("models", "flownet2-s.npz")
    height, width = 384, 768
    gen = FixedSizeFlowGenerator(trained_model_path, height, width)

    # Read and pre-process frames
    pre_processed_frames = [
        gen.pre_process_frame(cv2.imread(img_path)) for img_path in frame_paths
    ]
    num_frames = len(pre_processed_frames)
    num_flow_fields = num_frames - 1

    # Generate flow fields
    flow_fields = np.zeros(
        [num_flow_fields, height, width, 2], dtype=np.float32)
    for i in range(num_flow_fields):
        # Get frames
        left, right = pre_processed_frames[i], pre_processed_frames[i + 1]

        # Generate flow field
        flow_fields[i] = gen.apply(left, right)

    for i in range(num_flow_fields):
        # Get frames
        left, right = pre_processed_frames[i], pre_processed_frames[i + 1]

        # Visualize frames and flow field
        rgb_background = left[..., ::-1]
        fig = plot_2D_flow_field(flow_fields[i], rgb_background)
        plt.savefig("images/flow_field_{0}.png".format(i), bbox_inches="tight")
        # cv2.imshow("images", np.hstack([left, right]))
        # plt.show()
        plt.close(fig)
        # cv2.destroyAllWindows()

    # Smooth the flow fields across frames
    from scipy import ndimage

    for sigma in [0.15, 0.35, 0.5, 0.75, 1.0]:
        flow_fields = ndimage.gaussian_filter1d(flow_fields, 0.5, 0)
        for i in range(num_flow_fields):
            left = pre_processed_frames[i]
            rgb_background = left[..., ::-1]
            fig = plot_2D_flow_field(flow_fields[i], rgb_background)
            plt.savefig(
                "images/smooth_sigma_{0}_flow_field_{1}.png".format(sigma, i),
                bbox_inches="tight",
            )
            plt.close(fig)
