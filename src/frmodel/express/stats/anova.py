from typing import Dict
from frmodel.data.load import load_spec
from frmodel.base.D2.frame2D import Frame2D
import numpy as np
import scipy
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

class Anova(object):

    def __init__(self, g: Frame2D, bounds_file: str = '', factor: float = 1.0):
        self.features_ = []
        self.channel_names_  = {value:key for key, value in g.labels.items()}
        self.factor_ = factor
        self.trees_ = []

        ### -------------------------- Some parsing jives -------------------------------- ###
        assert bounds_file != ''
        self.bounds_ = pd.read_csv(bounds_file, header=None)
        self.bounds_['x1'] = self.bounds_.apply(lambda x: int(self.factor_ * int(x[0].split('|')[1])), axis=1)
        self.bounds_['x2'] = self.bounds_.apply(lambda x: int(self.factor_ * int(x[0].split('|')[2])), axis=1)
        self.bounds_['y1'] = self.bounds_.apply(lambda x: int(self.factor_ * int(x[0].split('|')[3])), axis=1)
        self.bounds_['y2'] = self.bounds_.apply(lambda x: int(self.factor_ * int(x[0].split('|')[4])), axis=1)
        self.bounds_[0] = self.bounds_.apply(lambda x: x[0].split('|')[0], axis=1)
        self.bounds_ = self.bounds_.rename(columns={0: 'species'})
    
        ### ----------------- Segment the trees from the meta Frame2D ---------------------- ###
        for i in range(len(self.bounds_)):
            tree = g[self.bounds_.loc[i, 'x1']:self.bounds_.loc[i, 'x2'], self.bounds_.loc[i, 'y1']:self.bounds_.loc[i, 'y2']]
            self.trees_.append(tree)

    def aggregate_features(self):
        feature_names = []
        for tree in self.trees_:
            features = np.zeros([tree.data.shape[-1] * 4])

            ### -------- Compute the features from the different GLCM features --------- ###
            for i in range(tree.data.shape[-1]):
                # Create an empty element of rows -> size of channels, columns -> features
                channel = tree.data[:,:,i]
                # Reshape the image -> flatten it and individually compute individual features
                channel = channel.reshape(-1, 1)

                features[0 + (i * 4)] = np.nanmean(channel, axis=0)[0]
                features[1 + (i * 4)] = np.nanvar(channel, axis=0)[0]
                features[2 + (i * 4)] = scipy.stats.skew(channel, nan_policy='omit')[0]
                features[3 + (i * 4)] = scipy.stats.kurtosis(channel, nan_policy='omit')[0]

            self.features_.append(features)

        for i in range(self.trees_[0].data.shape[-1]):
            dict_ = self.channel_names_
            feature_names.append(dict_[i].lower() + "_mean")
            feature_names.append(dict_[i].lower() + "_var")
            feature_names.append(dict_[i].lower() + "_skew")
            feature_names.append(dict_[i].lower() + "_kurtosis")

        ### -------- Create features_ DataFrame. The DataFrame will contain the feature vector ------- ###
        self.features_ = pd.DataFrame(np.array(self.features_))
        self.features_.columns = feature_names
        self.features_['species'] = self.bounds_['species']
        return feature_names

    def anova(self, feature_names: list):
        independent_var = " + ".join(feature_names)

        # Add the dependent variable
        manova_formula = independent_var + ' ~ species'
        print(manova_formula)

        # Create MANOVA object
        maov = MANOVA.from_formula(manova_formula, data=self.features_)

        # Perform MANOVA
        print()
        print(maov.mv_test())