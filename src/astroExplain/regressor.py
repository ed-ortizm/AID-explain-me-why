import numpy as np
###############################################################################
class GalaxyPlus:
    """
    Class to add all pixel values in an image
    """
    def __init__(self):
        pass

    def predict(self, image: np.array)->float:
        # normalize pixels' space
        image *= 1/image.max()

        # normalize addition
        prediction = np.sum(image)/image.size

        return prediction
