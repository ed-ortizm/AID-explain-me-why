from configparser import ConfigParser, ExtendedInterpolation
import time
from PIL import Image
import pickle
import os

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
from skimage.segmentation import mark_boundaries

from astroExplain.regressor import GalaxyPlus

###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("imageExplain.ini")
###############################################################################
# Load image as np.array
print("Load galaxy image", end="\n")

input_directory = parser.get("directory", "input")
file_name = parser.get("file", "galaxy")
name_galaxy, in_format = file_name.split(".")

if in_format == "npy":

    galaxy = np.load(f"{input_directory}/{file_name}")

else:
    galaxy = np.array(
        Image.open(f"{input_directory}/{file_name}"),
        dtype=float
    )
# normalize pixel space
galaxy *= 1 / galaxy.max()

###############################################################################
output_directory = parser.get("directory", "output")
# Load model
addGalaxy = GalaxyPlus()

# Set explainer instance
explainer = lime_image.LimeImageExplainer(random_state=0)

for segmentation in ["slic", "quickshift", "felzenszwalb"]:

    segmentation_fn = SegmentationAlgorithm('quickshift')

    print(f"Explain with segmentation: {segmentation}", end="\n")
    # get explanation

    explanation = explainer.explain_instance(
        image=galaxy,
        classifier_fn=addGalaxy.predict,
        labels=None,
        hide_color=0,
        top_labels=1,
        # num_features=1000, # default= 100000
        num_samples=1_000,
        batch_size=10,
        segmentation_fn =segmentation_fn
        # distance_metric="cosine",
    )

    print(f"Finish explanation... Saving...", end="\n")

    save_name = f"{name_galaxy}Explanation{segmentation.capitalize()}"

    with open(f"{output_directory}/{save_name}.pkl", "wb") as file:

        pickle.dump(explanation, file)
###############################################################################
print("Inpect explanation", end="\n")

# save image with positive contributing meaningful features
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    num_features=100,
    hide_rest=False,
   # min_weight = -100
)

galaxyExplanation = mark_boundaries(temp, mask)
#import matplotlib.pyplot as plt
#plt.imshow(galaxyExplanation)
#plt.show()

###############################################################################
save_to = parser.get("directory", "output")
if os.path.exists(save_to) is False:
    os.mkdir(save_to)
np.save(f"{save_to}/{name_galaxy}_allExp.npy", galaxyExplanation)

# heatmap visualization
# code from lime repo
print("Get heat map visualization", end="\n")

#Select the same class explained on the figures above.
ind = explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
#plt.show()
plt.savefig(f"{save_to}/{name_galaxy}_heatMap.png")
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
