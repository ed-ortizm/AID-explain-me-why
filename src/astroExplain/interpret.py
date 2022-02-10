import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.lime_image import ImageExplanation
###############################################################################
class TellMeWhy:
    def __init__(self, explanation: ImageExplanation):

        self.explanation = explanation
        self.galaxy = explanation.image
        self.segments = explanation.segments

    ###########################################################################
    def show_me(
        self,
        positive_only: bool,
        negative_only: bool,
        number_of_features: int,
        hide_rest: bool,
        min_weight: float,
    ) -> np.array:

        image, mask = self.explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=num_features,
            hide_rest=hide_rest,
            min_weight = min_weight
        )

        visual_explanation = mark_boundaries(temp, mask)

        return visual_explanation
    ###########################################################################
    def heatmap(self,
        show_map: bool=False,
        save_map: bool=False,
        save_to: str=".", galaxy_name: str="name", save_format: str=".png"
    ) -> None:

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

        #The visualization makes more sense if a symmetrical colorbar is used.
        plt.imshow(
            heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max()
        )

        plt.colorbar()

        if show_map is True:
            plt.show()

        if save_map is True:

            plt.savefig(f"{save_to}/{galaxy_name}HeatMapExp.{save_format}")

        plt.close()
###############################################################################
