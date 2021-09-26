from typing import Dict
from frmodel.data.load import load_spec
from frmodel.base.D2.frame2D import Frame2D
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
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
        self.color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                      for i in range(len(self.bounds_))]

        ### ----------------- Segment the trees from the meta Frame2D ---------------------- ###
        for i in range(len(self.bounds_)):
            tree = g[self.bounds_.loc[i, 'x1']:self.bounds_.loc[i, 'x2'], self.bounds_.loc[i, 'y1']:self.bounds_.loc[i, 'y2']]
            self.trees_.append(tree)

    def display_distribution(self):
        """
        Displays the distributions and super imposes them onto one single seaborn plot
        """

        for j in range(self.trees_[0].data.shape[-1]):
            fig, ax = plt.subplots()

            for i, tree in enumerate(self.trees_):
                print(tree.data[:,:,j].shape)
                sns.distplot(tree.data[:,:,j], kde=False, color=self.color[i], ax=ax)
            
            plt.title(self.channel_names_[j])
            plt.show()    

    def aggregate_features(self):
        feature_names = []
        for tree in self.trees_:
            feature_vector_size = tree.data.shape[-1]
            features = np.zeros([feature_vector_size * 4])

            ### -------- Compute the features from the different GLCM features --------- ###
            for i in range(feature_vector_size):
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

    def aggregate_glcm(self):
        feature_vector_size = 0
        channel_count = 0

        for tree in self.trees_:
            channel_count = tree.data.shape[-1]
            for channel_index in range(channel_count):
                channel = tree.data[:,:,channel_index]
            feature_vector_size += channel.reshape(-1, 1).shape[0]
        
        # Initialise empty numpy vector 
        # + 1 to indicate the species name
        self.features_glcm_ = np.zeros([feature_vector_size, channel_count])
        species_name = []
        entry_last = 0
        entry_num = 0

        for tree_index, tree in enumerate(self.trees_):
            for channel_index in range(channel_count):
                channel = tree.data[:,:,channel_index]
                channel = channel.reshape(-1, 1)
                entry_num = entry_last
                for i in range(channel.shape[0]):
                    self.features_glcm_[entry_num, channel_index] = channel[i]
                    entry_num += 1

            for i in range(channel.shape[0]):
                species_name.append(self.bounds_['species'][tree_index])
            entry_last += channel.shape[0]

        self.features_glcm_ = pd.DataFrame(self.features_glcm_)
        channels = [self.channel_names_[i] for i in range(len(self.channel_names_))]
        print(channels)
        self.features_glcm_.columns = channels
        self.features_glcm_['species'] = species_name

    def anova(self, feature_name: str):
        assert self.features_glcm_ is not None

        print('\nANOVA for feature', feature_name)

        # Create ANOVA backbone table
        data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
        anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
        anova_table.set_index('Source of Variation', inplace = True)

        # calculate SSTR and update anova table
        x_bar = self.features_glcm_[feature_name].mean()
        SSTR = self.features_glcm_.groupby('species').count() * (self.features_glcm_.groupby('species').mean() - x_bar)**2
        anova_table['SS']['Between Groups'] = SSTR[feature_name].sum()

        # calculate SSE and update anova table
        SSE = (self.features_glcm_.groupby('species').count() - 1) * self.features_glcm_.groupby('species').std()**2
        anova_table['SS']['Within Groups'] = SSE[feature_name].sum()

        # calculate SSTR and update anova table
        SSTR = SSTR[feature_name].sum() + SSE[feature_name].sum()
        anova_table['SS']['Total'] = SSTR

        # update degree of freedom
        anova_table['df']['Between Groups'] = self.features_glcm_['species'].nunique() - 1
        anova_table['df']['Within Groups'] = self.features_glcm_.shape[0] - self.features_glcm_['species'].nunique()
        anova_table['df']['Total'] = self.features_glcm_.shape[0] - 1

        # calculate MS
        anova_table['MS'] = anova_table['SS'] / anova_table['df']

        # calculate F 
        F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
        anova_table['F']['Between Groups'] = F

        # p-value
        anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

        # F critical 
        alpha = 0.05
        # possible types "right-tailed, left-tailed, two-tailed"
        tail_hypothesis_type = "two-tailed"
        if tail_hypothesis_type == "two-tailed":
            alpha /= 2
        anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

        # Final ANOVA Table
        print(anova_table)

    def anova_stat_descriptors(self, feature_names: list):
        independent_var = " + ".join(feature_names)

        # Add the dependent variable
        manova_formula = independent_var + ' ~ species'
        print(manova_formula)

        # Create MANOVA object
        maov = MANOVA.from_formula(manova_formula, data=self.features_)

        # Perform MANOVA
        print()
        print(maov.mv_test())