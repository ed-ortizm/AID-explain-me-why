from configparser import ConfigParser, ExtendedInterpolation
import time
from PIL import Image

import numpy as np

from astroExplain.regressor import GalaxyPlus
###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("imageExplain.ini")
###############################################################################
# Load image as np.array

img = np.array(Image.open("data/NGC3432.jpg")).astype(float)

# Load model
galaxyAdd = GalaxyPlus()
print(galaxyAdd.predict(img))
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
