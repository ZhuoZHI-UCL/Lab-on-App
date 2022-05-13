#All the library used
import os
import joblib
import keras
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from numpy import concatenate
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets,decomposition,manifold
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets,decomposition,manifold
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets,decomposition,manifold
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import  svm
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from keras.callbacks import CSVLogger
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso,BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor, XGBRFClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale,minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score#R squar
import cmath
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy import stats
import math
import argparse
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score

#setting for plotting
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100