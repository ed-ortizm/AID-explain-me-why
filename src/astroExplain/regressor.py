import numpy as np

###############################################################################
class GalaxyPlus:
    """
    Class to add all pixel values in an image
    """

    def __init__(self):
        pass

    def predict(self, image: np.array) -> np.array:

        image = self._update_dimension(image)

        # predict and normalize
        prediction = np.sum(image, axis=(1,2,3)) #/ image[0, :].size
        # print(prediction.shape)

        return prediction.reshape((-1,1))

    def _update_dimension(self, image:np.array) -> np.array:

        if image.ndim == 3:
            return image[np.newaxis, ...]

        return image
