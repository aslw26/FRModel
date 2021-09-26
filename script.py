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
# g = Frame2D.load("glcm.npz")
f = Frame2D.from_image("Original.png")
print('Loaded')

# analysis_instance = Anova(g, 'bounds.csv')
#analysis_instance.display_distribution()
#feature_names = analysis_instance.aggregate_features()
# analysis_instance.aggregate_glcm()
fpl = f.plot()
fig = fpl.image()
fig.savefig('out.jpg')
#  R, G, B, X, Y, H, S, V, EX_G, MEX_G, EX_GR, NDI, VEG,
#  ConR, ConG, ConB, CorrR, CorrG, CorrB, EntR, EntG, EntB
# analysis_instance.anova('RED')
# analysis_instance.anova('GREEN')
# analysis_instance.anova('BLUE')
# analysis_instance.anova('RED_EDGE')
# analysis_instance.anova('NIR')
# analysis_instance.anova('CON_RED')
# analysis_instance.anova('CON_GREEN')
# analysis_instance.anova('CON_BLUE')
# analysis_instance.anova('CON_RED_EDGE')
# analysis_instance.anova('CON_NIR')
# analysis_instance.anova('COR_RED')
# analysis_instance.anova('COR_GREEN')
# analysis_instance.anova('COR_BLUE')
# analysis_instance.anova('COR_RED_EDGE')
# analysis_instance.anova('COR_NIR')
# analysis_instance.anova('ASM_RED')
# analysis_instance.anova('ASM_GREEN')
# analysis_instance.anova('ASM_BLUE')
# analysis_instance.anova('ASM_RED_EDGE')
# analysis_instance.anova('ASM_NIR')
# analysis_instance.anova('MEAN_RED')
# analysis_instance.anova('MEAN_GREEN')
# analysis_instance.anova('MEAN_BLUE')
# analysis_instance.anova('MEAN_RED_EDGE')
# analysis_instance.anova('MEAN_NIR')
# analysis_instance.anova('VAR_RED')
# analysis_instance.anova('VAR_GREEN')
# analysis_instance.anova('VAR_BLUE')
# analysis_instance.anova('VAR_RED_EDGE')
# analysis_instance.anova('VAR_NIR')