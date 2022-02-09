from configparser import ConfigParser, ExtendedInterpolation
import time
from PIL import Image

from lime import lime_image
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
name_galaxy = parser.get("file", "galaxy")

galaxy = np.array(Image.open("data/NGC3432.jpg")).astype(float)

###############################################################################
print("Explain GalaxyPlus model", end="\n")
# Load model
addGalaxy = GalaxyPlus()
# Set explainer instance
explainer = lime_image.LimeImageExplainer()
# get explanation
explanation = explainer.explain_instance(
    image=galaxy,
    classifier_fn=addGalaxy.predict,
    # top_labels=5,
    hide_color=0,
    num_samples=100
)
###############################################################################
print("Inpect explanation", end="\n")
save_to = parser.get("directory", "output")
# save image with positive contributing meaningful features
temp, mask = explanation.get_image_and_mask(
    # explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=True
)

galaxyExplanation = mark_boundaries(temp / 2 + 0.5, mask)
np.save(f"{save_to}/{name_galaxy}_positive.npy", galaxyExplanation)
###########################################################
# save whole image with boundaries
temp, mask = explanation.get_image_and_mask(
    # explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

galaxyExplanation = mark_boundaries(temp / 2 + 0.5, mask)
np.save(f"{save_to}/{name_galaxy}_positiveBounded.npy", galaxyExplanation)
###########################################################
# save represantions ilustrating possitive and detrimental
# menainful features

temp, mask = explanation.get_image_and_mask(
    # explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

galaxyExplanation = mark_boundaries(temp / 2 + 0.5, mask)
np.save(f"{save_to}/{name_galaxy}_contribute.npy", galaxyExplanation)
###############################################################################
# heatmap visualization
# code from lime repo
print("Get heat map visualization", end="\n")

#Select the same class explained on the figures above.
ind =  explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()

plt.savefig(f"{save_to}/{name_galaxy}_heatMap.png")
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
