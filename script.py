from frmodel.data.load import load_spec
from frmodel.base.D2.frame2D import Frame2D
from frmodel.express.stats.anova import Anova
import numpy as np
import scipy
import pandas as pd
from PIL import Image

dir_path = 'rsc/imgs/spec/chestnut/18Dec2020/'
scale = 0.2
factor = 1


# Load the file
print('Loading GLCM')
g = Frame2D.load("glcm_small.npz")
print('Loaded')

analysis_instance = Anova(g, 'bounds.csv', 0.05)
feature_names = analysis_instance.aggregate_features()
analysis_instance.anova(feature_names)