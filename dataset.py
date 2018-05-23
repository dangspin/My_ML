#! /usr/bin/python

"""
This file is used to load the dataset for machine learning algorithm calculations

Author: Xiaoqian Dang

"""

import numpy as np
import pandas as pd


class load_watermelon:
    """
    watermelon dataset
    """

    def __init__(self):

        self.df = pd.read_csv("watermelon.csv", delimiter = ",", header = None)
        self.data = self.df.values[:,0:8]
        self.target = self.df.values[:,8]
        self.num_of_sample = self.data.shape[0]
        self.num_of_feature = self.data.shape[1]
        self.feature_names = ['color','root','sound','texture','bottom','touch','density','suger_rate']
        self.meanings = '''
        color:   has values 1-3; 1: white; 2: green;     3: black.
        root:    has values 1-3; 1: curl;  2: very curl; 3: straight hard
        sound:   has values 1-3; 1: clear; 2: not clear; 3: no sound
        texture: has values 1-3; 1: clear; 2: dim;       3: very dim
        bottom:  has values 1-3; 1: flat;  2: not flat   3: dent
        touch:   has values 1-2; 1: soft;  2: hard
        '''

    def get_meanings(self):
        """
        This function is used to format print the meanings of each feature.
        """
        print (self.meanings)


## Here I use the dataset about loan.
## The basic useage of this class is the same as load_iris in sklearn

class load_loan:

    def __init__(self):

        self.df = pd.read_csv("loan.csv", delimiter = ",", header = None)
        self.data = self.df.values[:,0:4]
        self.target = self.df.values[:,4]
        self.num_of_sample = self.data.shape[0]
        self.num_of_feature = self.data.shape[1]
        self.feature_names = ['age','has_job','has_house','credit']
        self.meanings = '''
        age:        has values 0-2; 0: youth; 1: adult; 2: old.
        has_job:    has values 0-1; 0: no;    1: yes;
        has_house:  has values 0-1; 0: no;    1: yes;
        credit:     has values 0-2; 0: low;   1: good;  2: very good
        '''

    def get_meanings(self):

        print (self.meanings)
