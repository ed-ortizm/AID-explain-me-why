from configparser import ConfigParser, ExtendedInterpolation
import time
from PIL import Image

from lime import lime_image
import numpy as np

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
galaxyAdd = GalaxyPlus()
print(galaxyAdd.predict(galaxy))
# Set explainer instance
explainer = lime_image.LimeImageExplainer()
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
