from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import time
###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("imageExplain.ini")

###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
